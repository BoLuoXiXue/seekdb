/*
 * Copyright (c) 2026 OceanBase.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "hipvs_build.h"

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/hnsw.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void check(cuvsError_t result, const char *operation)
{
  if (result != CUVS_SUCCESS) {
    const char *detail = cuvsGetLastErrorText();
    throw std::runtime_error(std::string(operation) + ": " + (detail ? detail : "hipVS error"));
  }
}

struct Resources
{
  cuvsResources_t resources = 0;
  cuvsCagraIndexParams_t params = nullptr;
  cuvsCagraIndex_t graph = nullptr;
  cuvsIvfPqSearchParams_t pq_search = nullptr;
  cuvsHnswIndexParams_t hnsw_params = nullptr;
  cuvsHnswIndex_t hnsw = nullptr;
  ~Resources()
  {
    if (hnsw) { cuvsHnswIndexDestroy(hnsw); }
    if (hnsw_params) { cuvsHnswIndexParamsDestroy(hnsw_params); }
    if (graph) { cuvsCagraIndexDestroy(graph); }
    if (params) { cuvsCagraIndexParamsDestroy(params); }
    if (pq_search) { cuvsIvfPqSearchParamsDestroy(pq_search); }
    if (resources) { cuvsResourcesDestroy(resources); }
  }
};

// hnswlib uses native little-endian 64-bit size_t and labeltype on this platform.
struct HnswHeader
{
  uint64_t offset_level0, max_elements, count, stride, label_offset, data_offset;
  int32_t max_level;
  uint32_t entrypoint;
  uint64_t max_m, max_m0, m;
  double multiplier;
  uint64_t ef_construction;
};
static_assert(sizeof(HnswHeader) == 96, "unsupported hnswlib ABI");

class MappedFile
{
public:
  explicit MappedFile(const char *path)
  {
    fd_ = open(path, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    struct stat info{};
    if (fd_ < 0 || fstat(fd_, &info) != 0 || info.st_size < sizeof(HnswHeader)) {
      if (fd_ >= 0) { close(fd_); }
      throw std::runtime_error("cannot read serialized HNSW");
    }
    size_ = info.st_size;
    data_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
      close(fd_);
      throw std::runtime_error("cannot map serialized HNSW");
    }
  }
  ~MappedFile() { munmap(data_, size_); close(fd_); }
  char *data() { return static_cast<char *>(data_); }
  uint64_t size() const { return size_; }
private:
  int fd_;
  void *data_;
  uint64_t size_;
};

void remap_ids(const char *path, const float *vectors, const int64_t *ids, int64_t count, int dim,
    int max_degree, HipvsBuildStats &stats)
{
  MappedFile file(path);
  HnswHeader h;
  std::memcpy(&h, file.data(), sizeof(h));
  if (h.offset_level0 != 0 || h.count != static_cast<uint64_t>(count)
      || h.max_elements < h.count || h.max_m != static_cast<uint64_t>(max_degree)
      || h.m != h.max_m || h.max_m0 != 2 * h.m
      || h.data_offset != sizeof(uint32_t) * (h.max_m0 + 1)
      || h.label_offset != h.data_offset + static_cast<uint64_t>(dim) * sizeof(float)
      || h.stride != h.label_offset + sizeof(int64_t) || h.max_level < 0
      || h.entrypoint >= h.count || h.stride > (file.size() - sizeof(h)) / h.count) {
    throw std::runtime_error("hipVS HNSW format is incompatible with VSAG");
  }
  std::vector<uint8_t> seen(count, 0);
  for (uint64_t row = 0; row < h.count; ++row) {
    char *record = file.data() + sizeof(h) + row * h.stride;
    uint32_t degree;
    std::memcpy(&degree, record, sizeof(degree));
    if (degree > h.max_m0) { throw std::runtime_error("invalid HNSW neighbor count"); }
    for (uint32_t j = 0; j < degree; ++j) {
      uint32_t neighbor;
      std::memcpy(&neighbor, record + sizeof(uint32_t) * (j + 1), sizeof(neighbor));
      if (neighbor >= h.count) { throw std::runtime_error("invalid HNSW neighbor ID"); }
    }
    uint64_t original;
    std::memcpy(&original, record + h.label_offset, sizeof(original));
    if (original >= h.count || seen[original]++) {
      throw std::runtime_error("invalid HNSW row mapping");
    }
    if (std::memcmp(record + h.data_offset, vectors + original * dim,
        static_cast<size_t>(dim) * sizeof(float)) != 0) {
      throw std::runtime_error("serialized HNSW changed an input FP32 vector");
    }
    std::memcpy(record + h.label_offset, &ids[original], sizeof(int64_t));
  }
  stats.rows = h.count;
  stats.hierarchy_levels = h.max_level + 1;
}
} // namespace

extern "C" int seekdb_hipvs_build_v1(const float *vectors, const int64_t *ids,
    int64_t count, int32_t dim, int32_t max_degree, int32_t ef_construction,
    const char *output, HipvsBuildStats *stats, char *error, uint64_t error_capacity)
{
  try {
    if (!vectors || !ids || !output || !stats || count < 256 || dim <= 0
        || count > std::numeric_limits<uint32_t>::max() || max_degree < 2
        || max_degree > 256 || ef_construction < max_degree) {
      throw std::invalid_argument("unsupported hipVS HNSW build parameters");
    }
    std::vector<int64_t> unique_ids(ids, ids + count);
    std::sort(unique_ids.begin(), unique_ids.end());
    if (std::adjacent_find(unique_ids.begin(), unique_ids.end()) != unique_ids.end()) {
      throw std::invalid_argument("duplicate vector IDs in snapshot build");
    }
    *stats = {};
    Resources r;
    check(cuvsResourcesCreate(&r.resources), "create resources");
    check(cuvsDeviceIdGet(r.resources, &stats->device), "get GPU device");
    check(cuvsCagraIndexParamsCreate(&r.params), "create CAGRA parameters");
    r.params->metric = L2Expanded;
    r.params->graph_degree = 2 * max_degree;
    r.params->intermediate_graph_degree = std::min<int64_t>(count - 1,
        std::max<int64_t>(2 * r.params->graph_degree, ef_construction));
    if (count >= 16384) {
      // GPU IVF-PQ proposes graph candidates; refinement uses the complete
      // original FP32 vectors. Neither the persisted vectors nor CPU search
      // use PQ codes. Increasing refinement fixes the low-degree graph recall
      // loss without increasing the SQL M parameter or query ef_search.
      r.params->build_algo = IVF_PQ;
      check(cuvsIvfPqSearchParamsCreate(&r.pq_search), "create graph candidate parameters");
      r.pq_search->n_probes = 32;
      r.pq_search->lut_dtype = HIP_R_32F;
      r.pq_search->internal_distance_dtype = HIP_R_32F;
      // CAGRA owns this outer structure; the search parameters remain ours.
      r.params->graph_build_params = new cuvsIvfPqParams{nullptr, r.pq_search, 4.0f};
    } else {
      // At this size, constructing the coarse IVF index costs more than the
      // graph work it saves. NN-Descent avoids that fixed startup overhead.
      r.params->build_algo = NN_DESCENT;
      r.params->intermediate_graph_degree = std::min<int64_t>(count - 1,
          std::max<int64_t>(256, r.params->intermediate_graph_degree));
    }
    r.params->nn_descent_niter = 20;
    check(cuvsCagraIndexCreate(&r.graph), "create CAGRA index");
    int64_t shape[] = {count, dim};
    DLManagedTensor dataset{};
    dataset.dl_tensor.data = const_cast<float *>(vectors);
    dataset.dl_tensor.device = {kDLCPU, 0};
    dataset.dl_tensor.ndim = 2;
    dataset.dl_tensor.dtype = {kDLFloat, 32, 1};
    dataset.dl_tensor.shape = shape;
    auto start = Clock::now();
    check(cuvsCagraBuild(r.resources, r.params, &dataset, r.graph), "GPU graph build");
    check(cuvsStreamSync(r.resources), "synchronize GPU graph");
    stats->graph_seconds = seconds(start);

    check(cuvsHnswIndexParamsCreate(&r.hnsw_params), "create HNSW parameters");
    r.hnsw_params->hierarchy = GPU;
    r.hnsw_params->ef_construction = ef_construction;
    r.hnsw_params->num_threads = 1;
    check(cuvsHnswIndexCreate(&r.hnsw), "create HNSW index");
    start = Clock::now();
    check(cuvsHnswFromCagraWithDataset(r.resources, r.hnsw_params, r.graph, r.hnsw,
        &dataset), "GPU HNSW hierarchy build");
    check(cuvsStreamSync(r.resources), "synchronize HNSW hierarchy");
    stats->hierarchy_seconds = seconds(start);

    start = Clock::now();
    check(cuvsHnswSerialize(r.resources, output, r.hnsw), "serialize HNSW");
    remap_ids(output, vectors, ids, count, dim, max_degree, *stats);
    stats->serialize_seconds = seconds(start);
    return 0;
  } catch (const std::exception &e) {
    if (error && error_capacity) { std::snprintf(error, error_capacity, "%s", e.what()); }
    return -1;
  } catch (...) {
    if (error && error_capacity) { std::snprintf(error, error_capacity, "unknown hipVS failure"); }
    return -1;
  }
}
