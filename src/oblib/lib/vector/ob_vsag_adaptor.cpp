/*
 * Copyright (c) 2025 OceanBase.
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

#define USING_LOG_PREFIX LIB

#include "ob_vsag_adaptor.h"
#include "hipvs_build_buffer.h"
#include <map>
#include <cstdio>
#include <cstring>
#include <fstream>
#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif
#include "vsag/vsag.h"
#include "vsag/constants.h"
#include "vsag/errors.h"
#include "vsag/dataset.h"
#include "vsag/search_param.h"
#include "vsag/index.h"
#include "vsag/options.h"
#include "vsag/factory.h"
#include "lib/utility/ob_print_utils.h"
#include "lib/oblog/ob_log.h"
#include "lib/worker.h"

namespace oceanbase {
namespace common {
namespace obvsag {

using namespace vsag;

static constexpr int64_t VSAG_SINDI_N_CANDIDATE_FACTOR = 500;

static int vsag_errcode2ob(vsag::ErrorType vsag_errcode)
{
  int ret = OB_ERR_VSAG_RETURN_ERROR;
  switch (vsag_errcode) {
    case vsag::ErrorType::INVALID_ARGUMENT: {
      ret = OB_INVALID_ARGUMENT;
      break;
    }
    case vsag::ErrorType::UNSUPPORTED_INDEX:
    case vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION: {
      ret = OB_NOT_SUPPORTED;
      break;
    }
    case vsag::ErrorType::DIMENSION_NOT_EQUAL: {
      ret = OB_INVALID_ARGUMENT;
      break;
    }
    case vsag::ErrorType::INDEX_EMPTY: {
      ret = OB_OP_NOT_ALLOW;
      break;
    }
    case vsag::ErrorType::NO_ENOUGH_MEMORY: {
      ret = OB_ALLOCATE_MEMORY_FAILED;
      break;
    }
    default: {
      ret = OB_ERR_VSAG_RETURN_ERROR;
      break;
    }
  }
  return ret;
}

static void fill_vsag_error_message(const vsag::Error &error, std::string &err_msg)
{
  err_msg = error.message;
}

static void adjust_create_index_max_degree(const IndexType index_type, int &max_degree)
{
  // hgraph of vsag needs to be multiplied by 2 so as to align recall with hnsw
  if (HNSW_SQ_TYPE == index_type || HNSW_BQ_TYPE == index_type || HGRAPH_TYPE == index_type) {
    max_degree *= 2;
    LOG_INFO("change max_degree for hgraph", K(index_type), K(max_degree), K(lbt()));
  }
}

class ObVasgFilter final : public vsag::Filter {
public:
  ObVasgFilter(float valid_ratio,
               const std::function<bool(int64_t)> &vid_fallback_func,
               const std::function<bool(const char *)> &exinfo_fallback_func)
      : valid_ratio_(valid_ratio), vid_fallback_func_(vid_fallback_func),
        exinfo_fallback_func_(exinfo_fallback_func){};

  ~ObVasgFilter() {}

  bool CheckValid(int64_t id) const override { return !vid_fallback_func_(id); }

  bool CheckValid(const char *data) const override {
    return !exinfo_fallback_func_(data);
  }

  float ValidRatio() const override { return valid_ratio_; }

private:
  float valid_ratio_;
  std::function<bool(int64_t)> vid_fallback_func_{nullptr};
  std::function<bool(const char *)> exinfo_fallback_func_{nullptr};
};

class HnswIndexHandler {
public:
  HnswIndexHandler(bool is_create, bool is_build, bool use_static,
                   const char *dtype, const char *metric, int max_degree,
                   int ef_construction, int ef_search, int dim,
                   IndexType index_type, std::shared_ptr<vsag::Index> index,
                   vsag::Allocator *allocator, uint64_t extra_info_size,
                   int16_t refine_type, int16_t bq_bits_query, bool bq_use_fht)
      : is_created_(is_create), is_build_(is_build), use_static_(use_static),
        dtype_(dtype), metric_(metric), max_degree_(max_degree),
        ef_construction_(ef_construction), ef_search_(ef_search), dim_(dim),
        index_type_(index_type), index_(index), allocator_(allocator),
        extra_info_size_(extra_info_size), refine_type_(refine_type),
        bq_bits_query_(bq_bits_query), bq_use_fht_(bq_use_fht) {}

  HnswIndexHandler(bool is_create, bool is_build, bool use_static, const char *dtype, const char *metric,
      IndexType index_type, std::shared_ptr<vsag::Index> index, vsag::Allocator *allocator, uint64_t extra_info_size,
      bool use_reorder, float doc_prune_ratio, int window_size)
      : is_created_(is_create),
        is_build_(is_build),
        use_static_(use_static),
        dtype_(dtype),
        metric_(metric),
        index_type_(index_type),
        index_(index),
        allocator_(allocator),
        extra_info_size_(extra_info_size),
        use_reorder_(use_reorder),
        doc_prune_ratio_(doc_prune_ratio),
        window_size_(window_size)
  {}

  ~HnswIndexHandler() {
    index_ = nullptr;
    LOG_INFO("[OBVSAG] after deconstruction, hnsw index", KP(allocator_), K(index_.use_count()), K(lbt()));
  }
  int build_index(const vsag::DatasetPtr &base);
  int get_index_number();
  int add_index(const vsag::DatasetPtr &incremental);
  int cal_distance_by_id(const float *vector, const int64_t *ids, int64_t count,
                         const float *&dist);
  int cal_distance_by_id(uint32_t len, uint32_t *dims, float *vals,
                        const int64_t *ids, int64_t count, const float *&dist);
  int get_extra_info_by_ids(const int64_t *ids, int64_t count,
                            char *extra_infos);
  int get_vid_bound(int64_t &min_vid, int64_t &max_vid);
  uint64_t estimate_memory(const uint64_t row_count, const bool is_build);
  int knn_search(const vsag::DatasetPtr &query, int64_t topk,
                 const std::string &parameters, const float *&dist,
                 const int64_t *&ids, int64_t &result_size, float valid_ratio,
                 int index_type, FilterInterface *bitmap, bool reverse_filter,
                 bool need_extra_info, const char *&extra_infos,
                 void *allocator, float distance_threshold = FLT_MAX);
  int knn_search(const vsag::DatasetPtr &query, int64_t topk,
                 const std::string &parameters, const float *&dist,
                 const int64_t *&ids, int64_t &result_size, float valid_ratio,
                 int index_type, FilterInterface *bitmap, bool reverse_filter,
                 bool need_extra_info, const char *&extra_infos,
                 void *&iter_ctx, bool is_last_search, void *allocator);
  int immutable_optimize();
  int enable_hipvs_build();
  int finish_hipvs_build();
  bool is_hipvs_built()
  {
    if (!hipvs_build_) { return false; }
    std::lock_guard<std::mutex> lock(hipvs_build_->mutex);
    return hipvs_build_->built_on_gpu;
  }

  std::shared_ptr<vsag::Index> &get_index() { return index_; }
  void set_index(std::shared_ptr<vsag::Index> hnsw)
  {
    if (hipvs_build_) {
      std::lock_guard<std::mutex> lock(hipvs_build_->mutex);
      index_ = hnsw;
      // A deserialized graph is already complete; do not mask its row count
      // with an empty build buffer or rebuild it on the next read.
      hipvs_build_->release();
      hipvs_build_->built_on_gpu = false;
    } else {
      index_ = hnsw;
    }
  }
  vsag::Allocator *get_allocator() const { return allocator_; }
  inline bool get_use_static() const { return use_static_; }
  inline int get_max_degree() const { return max_degree_; }
  inline int get_ef_construction() const { return ef_construction_; }
  inline int get_index_type() const { return (int)index_type_; }
  const char *get_dtype() const { return dtype_; }
  const char *get_metric() const { return metric_; }
  inline int get_ef_search() const { return ef_search_; }
  inline int get_dim() const { return dim_; }
  inline uint64_t get_extra_info_size() const { return extra_info_size_; }
  inline int16_t get_refine_type() const { return refine_type_; }
  inline int16_t get_bq_bits_query() const { return bq_bits_query_; }
  inline bool get_bq_use_fht() const { return bq_use_fht_; };
  inline bool get_use_reorder() const { return use_reorder_; }
  inline float get_doc_prune_ratio() const { return doc_prune_ratio_; }
  inline int get_window_size() const { return window_size_; }

  TO_STRING_KV(KP(this), K_(is_created), K_(is_build), K_(use_static), KCSTRING_(dtype),
      KCSTRING_(metric), K_(max_degree), K_(ef_construction), K_(ef_search), K_(dim),
      K_(ef_search), K_(index_type), KP(index_.get()), KP_(allocator), K_(extra_info_size),
      K_(refine_type), K_(bq_bits_query), K_(bq_use_fht));

private:
  bool is_created_;
  bool is_build_;
  bool use_static_;
  const char *dtype_;
  const char *metric_;
  int max_degree_;
  int ef_construction_;
  int ef_search_;
  int dim_;
  IndexType index_type_;
  std::shared_ptr<vsag::Index> index_;
  vsag::Allocator *allocator_;
  uint64_t extra_info_size_;
  int16_t refine_type_;
  int16_t bq_bits_query_;
  bool bq_use_fht_;
  bool use_reorder_;
  float doc_prune_ratio_;
  int window_size_;
  std::unique_ptr<HipvsBuildBuffer> hipvs_build_;
};

static const char *hipvs_worker_path()
{
  static const std::string path = []() -> std::string {
    const char *override = std::getenv("SEEKDB_HIPVS_BUILD_WORKER");
    if (override && *override) { return override; }
#if defined(__linux__)
    char executable[4096];
    const ssize_t size = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (size > 0 && size < sizeof(executable) - 1) {
      executable[size] = '\0';
      std::string directory(executable);
      directory.resize(directory.find_last_of('/') + 1);
      return directory + "seekdb_hipvs_build_worker";
    }
#endif
    return {};
  }();
  return path.empty() ? nullptr : path.c_str();
}

#if defined(__linux__)
static int run_hipvs_worker(const float *vectors, const int64_t *ids, int64_t count,
    int dim, int max_degree, int ef_construction, const char *output,
    HipvsBuildStats &stats, std::string &error)
{
  // File transport is part of the construction clock. Use a fresh worker for
  // each graph: allocator and hnswlib symbols cannot interpose with the server.
  const std::string input_path = std::string(output) + ".input";
  const std::string stats_path = std::string(output) + ".stats";
  const std::string log_path = std::string(output) + ".log";
  const auto remove_files = [&](const char *) {
    unlink(input_path.c_str()); unlink(stats_path.c_str()); unlink(log_path.c_str());
  };
  std::unique_ptr<const char, decltype(remove_files)> cleanup(output, remove_files);
  HipvsBuildInput header{HIPVS_BUILD_INPUT_MAGIC, count, dim, max_degree, ef_construction, 0};
  const int input_fd = open(input_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (input_fd < 0) { error = "cannot create private GPU worker input"; return -1; }
  close(input_fd);
  const int stats_fd = open(stats_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (stats_fd < 0) { error = "cannot create private GPU worker statistics"; return -1; }
  close(stats_fd);
  std::ofstream file(input_path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char *>(&header), sizeof(header));
  file.write(reinterpret_cast<const char *>(vectors), count * dim * sizeof(float));
  file.write(reinterpret_cast<const char *>(ids), count * sizeof(int64_t));
  file.close();
  if (!file) { error = "cannot write GPU worker input"; return -1; }
  const char *worker = hipvs_worker_path();
  if (!worker) { error = "GPU worker path is not configured"; return -1; }
  char *argv[] = {const_cast<char *>(worker), const_cast<char *>(input_path.c_str()),
      const_cast<char *>(output), const_cast<char *>(stats_path.c_str()), nullptr};
  posix_spawn_file_actions_t actions;
  int result = posix_spawn_file_actions_init(&actions);
  if (result) { error = std::strerror(result); return -1; }
  result = posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log_path.c_str(),
      O_CREAT | O_TRUNC | O_WRONLY, 0600);
  if (!result) { result = posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO); }
  pid_t pid = -1;
  if (!result) { result = posix_spawn(&pid, worker, &actions, nullptr, argv, ::environ); }
  posix_spawn_file_actions_destroy(&actions);
  if (result) { error = std::strerror(result); return -1; }
  int status = 0;
  pid_t waited;
  do { waited = waitpid(pid, &status, 0); } while (waited < 0 && errno == EINTR);
  if (waited != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::ifstream log(log_path);
    char text[2048] = {};
    log.read(text, sizeof(text) - 1);
    error = "worker status " + std::to_string(status) + ": " + text;
    return -1;
  }
  std::ifstream statistics(stats_path, std::ios::binary);
  if (!statistics.read(reinterpret_cast<char *>(&stats), sizeof(stats))
      || stats.rows != static_cast<uint64_t>(count)) {
    error = "incomplete GPU worker statistics"; return -1;
  }
  return 0;
}
#endif

int HnswIndexHandler::enable_hipvs_build()
{
  int ret = OB_SUCCESS;
  const char *enabled = std::getenv("SEEKDB_HIPVS_BUILD");
  const bool required = enabled && std::strcmp(enabled, "1") == 0;
  const bool automatic = !enabled || !*enabled || std::strcmp(enabled, "auto") == 0;
  if (enabled && *enabled && !required && !automatic && std::strcmp(enabled, "0") != 0) {
    ret = OB_INVALID_ARGUMENT;
    LOG_WARN("[HIPVS_BUILD] invalid SEEKDB_HIPVS_BUILD; expected auto, 0 or 1", K(ret));
  } else if (required || automatic) {
    if (index_type_ != HNSW_TYPE || std::strcmp(dtype_, "float32") != 0
        || std::strcmp(metric_, "l2") != 0 || extra_info_size_ != 0
        || max_degree_ < 2 || max_degree_ > 256) {
      LOG_INFO("[HIPVS_BUILD] CPU fallback: unsupported snapshot parameters",
          K_(index_type), KCSTRING_(metric), K_(extra_info_size));
    }
#if defined(__linux__)
    else if (!hipvs_worker_path() || access(hipvs_worker_path(), X_OK) != 0
        || (automatic && access("/dev/kfd", R_OK | W_OK) != 0)) {
#else
    else if (true) {
#endif
      if (required) {
        ret = OB_NOT_SUPPORTED;
        LOG_WARN("[HIPVS_BUILD] requested GPU worker unavailable; build rejected", K(ret));
      } else {
        LOG_INFO("[HIPVS_BUILD] CPU fallback: automatic GPU worker or device unavailable");
      }
    } else {
      try {
        hipvs_build_.reset(new HipvsBuildBuffer(allocator_));
      } catch (const std::bad_alloc &) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      }
    }
  }
  return ret;
}

int enable_hipvs_build(VectorIndexPtr &index_handler)
{
  return index_handler ? static_cast<HnswIndexHandler *>(index_handler)->enable_hipvs_build()
                       : OB_INVALID_ARGUMENT;
}

bool is_hipvs_built(VectorIndexPtr index_handler)
{
  return index_handler && static_cast<HnswIndexHandler *>(index_handler)->is_hipvs_built();
}

int HnswIndexHandler::finish_hipvs_build()
{
  int ret = OB_SUCCESS;
  if (hipvs_build_) {
    std::lock_guard<std::mutex> lock(hipvs_build_->mutex);
    if (hipvs_build_->pending) {
      try {
        const int64_t count = hipvs_build_->ids.size();
        // The pinned VSAG HNSW constructor clamps M to [8, 64] and its
        // construction breadth to at least that effective M. Use the same
        // effective values rather than silently building a different graph
        // for declarations below/above those limits.
        const int effective_m = std::min(64, std::max(8, max_degree_));
        const int effective_ef = std::max(effective_m, ef_construction_);
        if (count < 256 || count <= 2 * effective_m) {
          // NN-Descent needs a sufficiently large graph. Do not claim GPU work
          // for empty or small snapshots; use the original VSAG builder.
          if (count > 0) {
            auto dataset = vsag::Dataset::Make();
            dataset->NumElements(count)->Dim(dim_)->Ids(hipvs_build_->ids.data())
                ->Float32Vectors(hipvs_build_->vectors.data())->Owner(false);
            auto result = index_->Build(dataset);
            if (!result) { ret = vsag_errcode2ob(result.error().type); }
          }
          LOG_INFO("[HIPVS_BUILD] CPU fallback: snapshot too small for graph degree",
              K(count), K_(max_degree), K(ret));
        } else {
#if defined(__linux__)
          const char *tmp = std::getenv("TMPDIR");
          std::string name = std::string(tmp ? tmp : "/tmp") + "/seekdb-hipvs-XXXXXX";
          std::vector<char> path(name.begin(), name.end());
          path.push_back('\0');
          const int fd = mkstemp(path.data());
          if (fd < 0) {
            ret = OB_IO_ERROR;
          } else {
            close(fd);
            // Always unlink the private intermediate, including on exceptions.
            const auto remove_file = [](char *file) { unlink(file); };
            std::unique_ptr<char, decltype(remove_file)> cleanup(path.data(), remove_file);
            HipvsBuildStats stats{};
            std::string error;
            if (run_hipvs_worker(hipvs_build_->vectors.data(), hipvs_build_->ids.data(), count,
                dim_, effective_m, effective_ef, path.data(), stats, error) != 0) {
              ret = OB_ERR_VSAG_RETURN_ERROR;
              LOG_WARN("[HIPVS_BUILD] GPU build failed; no CPU fallback", K(ret), "error", error.c_str());
            } else {
              std::ifstream file(path.data(), std::ios::binary | std::ios::ate);
              const std::streamoff length = file.tellg();
              if (length <= 0 || !file) {
                ret = OB_IO_ERROR;
              } else {
                // Deserialize through the existing VSAG reader, preserving all
                // CPU querying, filtering, ID lookup and storage serialization.
                vsag::Binary binary;
                auto allocator = HipvsBufferAllocator<int8_t>(allocator_);
                int8_t *bytes = allocator.allocate(length);
                binary.data = std::shared_ptr<int8_t[]>(bytes,
                    [allocator, length](int8_t *p) mutable { allocator.deallocate(p, length); });
                binary.size = length;
                file.seekg(0);
                if (!file.read(reinterpret_cast<char *>(bytes), length)) {
                  ret = OB_IO_ERROR;
                } else {
                  vsag::BinarySet set;
                  set.Set(vsag::HNSW_DATA, binary);
                  auto loaded = index_->Deserialize(set);
                  if (!loaded) {
                    ret = vsag_errcode2ob(loaded.error().type);
                    LOG_WARN("[HIPVS_BUILD] VSAG import failed", K(ret), "error", loaded.error().message.c_str());
                  } else if (index_->GetNumElements() != count) {
                    ret = OB_ERR_UNEXPECTED;
                  } else {
                    hipvs_build_->built_on_gpu = true;
                    LOG_INFO("[HIPVS_BUILD] GPU HNSW build and VSAG import completed", K(count), K_(dim),
                        "device", stats.device, "hierarchy_levels", stats.hierarchy_levels,
                        "declared_m", max_degree_, K(effective_m), K(effective_ef),
                        "graph_seconds", stats.graph_seconds, "hierarchy_seconds", stats.hierarchy_seconds,
                        "serialize_seconds", stats.serialize_seconds, "cpu_fallback", false);
                  }
                }
              }
            }
          }
#else
          ret = OB_NOT_SUPPORTED;
#endif
        }
        if (OB_SUCC(ret)) { hipvs_build_->release(); }
      } catch (const std::bad_alloc &) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      } catch (const std::exception &e) {
        ret = OB_ERR_VSAG_RETURN_ERROR;
        LOG_WARN("[HIPVS_BUILD] finalization exception", K(ret), "error", e.what());
      }
    }
  }
  return ret;
}

int HnswIndexHandler::build_index(const vsag::DatasetPtr &base)
{
  int ret = OB_SUCCESS;
  if (hipvs_build_) {
    if (OB_SUCC(ret = add_index(base))) { ret = finish_hipvs_build(); }
    return ret;
  }
  try {
    tl::expected<std::vector<int64_t>, Error> result = index_->Build(base);
    if (result.has_value()) {
    } else {
      ret = vsag_errcode2ob(result.error().type);
    }
  } catch (const std::exception &e) {
    ret = OB_ERR_VSAG_RETURN_ERROR;
    LOG_WARN("[OBVSAG] exception caught in build_index", "what", e.what());
  } catch (...) {
    ret = OB_ERR_VSAG_RETURN_ERROR;
    LOG_WARN("[OBVSAG] unknown exception caught in build_index");
  }
  return ret;
}

int HnswIndexHandler::get_index_number()
{
  if (hipvs_build_) {
    std::lock_guard<std::mutex> lock(hipvs_build_->mutex);
    if (hipvs_build_->pending) { return hipvs_build_->ids.size(); }
  }
  return index_->GetNumElements();
}

int HnswIndexHandler::add_index(const vsag::DatasetPtr &incremental)
{
  int ret = OB_SUCCESS;
  if (hipvs_build_) {
    std::lock_guard<std::mutex> lock(hipvs_build_->mutex);
    if (hipvs_build_->pending) {
      const int64_t count = incremental->GetNumElements();
      if (count < 0 || incremental->GetDim() != dim_
          || (count > 0 && (!incremental->GetIds() || !incremental->GetFloat32Vectors()))
          || count > UINT32_MAX - hipvs_build_->ids.size()) {
        return OB_INVALID_ARGUMENT;
      }
      try {
        // Reserve both before appending, so an allocation failure cannot leave
        // the vector/ID buffers at different row counts. Allocations use the
        // same tenant allocator as the original index.
        if (hipvs_build_->vectors.capacity() < hipvs_build_->vectors.size() + count * dim_) {
          hipvs_build_->vectors.reserve(std::max(hipvs_build_->vectors.size() + count * dim_,
            hipvs_build_->vectors.capacity() * 2));
        }
        if (hipvs_build_->ids.capacity() < hipvs_build_->ids.size() + count) {
          hipvs_build_->ids.reserve(std::max(hipvs_build_->ids.size() + count, hipvs_build_->ids.capacity() * 2));
        }
        if (count > 0) {
          hipvs_build_->vectors.insert(hipvs_build_->vectors.end(), incremental->GetFloat32Vectors(),
              incremental->GetFloat32Vectors() + count * dim_);
          hipvs_build_->ids.insert(hipvs_build_->ids.end(), incremental->GetIds(), incremental->GetIds() + count);
        }
      } catch (const std::bad_alloc &) {
        ret = OB_ALLOCATE_MEMORY_FAILED;
      }
      return ret;
    }
  }
  tl::expected<std::vector<int64_t>, Error> result = index_->Add(incremental);
  if (result.has_value()) {
    LOG_DEBUG("add index success", K(get_index_number()));
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

int HnswIndexHandler::cal_distance_by_id(uint32_t len, uint32_t *dims, float *vals,
                                         const int64_t *ids, int64_t count,
                                         const float *&dist)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(finish_hipvs_build())) { return ret; }
  vsag::SparseVector sparse;
  sparse.len_ = len;
  sparse.ids_ = dims;
  sparse.vals_ = vals;
  DatasetPtr query = vsag::Dataset::Make();
  query->NumElements(1)->SparseVectors(&sparse)->Owner(false);
  float *dist_tmp = (float*)allocator_->Allocate(count * sizeof(float));
  if (OB_ISNULL(dist_tmp)) {
    ret = OB_ALLOCATE_MEMORY_FAILED;
  }
  // TODO(ningxin.ning): support CalcDistanceById in sparse vector
  for (int i = 0; i < count && OB_SUCC(ret); ++i) {
    // tl::expected<float, Error> result = index_->CalcDistanceById(query, ids[i]);
    // if (result.has_value()) {
    //   dist_tmp[i] = result.value();
    // } else {
    //   ret = vsag_errcode2ob(result.error().type);
    // }
    dist_tmp[i] = 0.1;
  }
  dist = dist_tmp;
  return ret;
}

int HnswIndexHandler::cal_distance_by_id(const float *vector,
                                         const int64_t *ids, int64_t count,
                                         const float *&dist)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(finish_hipvs_build())) { return ret; }
  tl::expected<DatasetPtr, Error> result = index_->CalDistanceById(vector, ids, count);
  if (result.has_value()) {
    result.value()->Owner(false);
    dist = result.value()->GetDistances();
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

int HnswIndexHandler::get_extra_info_by_ids(const int64_t *ids, int64_t count,
                                            char *extra_infos)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(finish_hipvs_build())) { return ret; }
  tl::expected<void, Error> result = index_->GetExtraInfoByIds(ids, count, extra_infos);
  if (result.has_value()) {
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

int HnswIndexHandler::get_vid_bound(int64_t &min_vid, int64_t &max_vid)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(finish_hipvs_build())) { return ret; }
  int64_t element_cnt = index_->GetNumElements();
  if (element_cnt == 0) {
  } else {
    tl::expected<std::pair<int64_t, int64_t>, Error> result = index_->GetMinAndMaxId();
    if (result.has_value()) {
      min_vid = result.value().first;
      max_vid = result.value().second;
    } else {
      ret = vsag_errcode2ob(result.error().type);
    }
  }
  return ret;
}

uint64_t HnswIndexHandler::estimate_memory(const uint64_t row_count, const bool is_build)
{
  
  uint64_t size = 0;
  if (IPIVF_TYPE == index_type_) {
    // TODO(ningxin.ning): use vsag EstimateMemory
    size += 2 * sizeof(int64_t) * row_count;
    // nonzero dim = 100
    size += 100 * row_count * sizeof(float) * 2;
    if (use_reorder_) {
      size *= 2;
    }
  } else {
    size = index_->EstimateMemory(row_count);
  }
  if (hipvs_build_ && is_build) {
    // Include geometric input buffers and the serialized import buffer in the
    // host-side build estimate. GPU allocations are separately bounded by HIP.
    size += row_count * (3 * (sizeof(float) * dim_ + sizeof(int64_t))
        + sizeof(uint32_t) * (2 * max_degree_ + 1));
  }
  if (HNSW_BQ_TYPE == index_type_ && is_build) {
    if (QuantizationType::SQ8 == refine_type_) {
      size += (row_count * dim_ * sizeof(uint8_t));
    } else {
      size += (row_count * dim_ * sizeof(float));
    }
  }
  return size;
}

int HnswIndexHandler::immutable_optimize()
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(finish_hipvs_build())) { return ret; }
  if (index_type_ == IPIVF_TYPE) {
    // TODO(ningxin.ning): support SetImmutable for sparse vector index
  } else {
    tl::expected<void, Error> res = index_->SetImmutable();
    if (res.has_value()) {
      LOG_INFO("[OBVSAG] set immutable success", KPC(this));
    } else {
      ret = vsag_errcode2ob(res.error().type);
      LOG_WARN("[OBVSAG] index set immutable error", K(ret), K(res.error().type));
    }
  }
  return ret;
}

int HnswIndexHandler::knn_search(const vsag::DatasetPtr &query, int64_t topk,
                                 const std::string &parameters,
                                 const float *&dist, const int64_t *&ids,
                                 int64_t &result_size, float valid_ratio,
                                 int index_type, FilterInterface *bitmap,
                                 bool reverse_filter, bool need_extra_info,
                                 const char *&extra_infos, void *allocator,
                                 float distance_threshold)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(finish_hipvs_build())) { return ret; }
  std::function<bool(int64_t)> vid_filter = [bitmap, reverse_filter](int64_t id) -> bool {
    if (!reverse_filter) {
      return bitmap->test(id);
    } else {
      return !(bitmap->test(id));
    }
  };
  std::function<bool(const char *)> exinfo_filter = [bitmap, reverse_filter](const char *data) -> bool {
    if (!reverse_filter) {
      return bitmap->test(data);
    } else {
      return !(bitmap->test(data));
    }
  };

  std::shared_ptr<ObVasgFilter> vsag_filter = std::make_shared<ObVasgFilter>(valid_ratio, vid_filter, exinfo_filter);
  vsag::Allocator *vsag_allocator = nullptr;
  if (allocator != nullptr) vsag_allocator = static_cast<vsag::Allocator *>(allocator);
  tl::expected<std::shared_ptr<vsag::Dataset>, vsag::Error> result;
  if (index_type_ == IPIVF_TYPE) {
    result = index_->KnnSearch(query, topk, parameters, bitmap == nullptr ? nullptr : vsag_filter);
  } else {
    vsag::SearchParam search_param(false, parameters,
                                 bitmap == nullptr ? nullptr : vsag_filter,
                                 vsag_allocator);
    result = index_->KnnSearch(query, topk, search_param);
  }
  if (result.has_value()) {
    // the lifecycle of result
    result.value()->Owner(false);
    ids = result.value()->GetIds();
    dist = result.value()->GetDistances();
    result_size = result.value()->GetDim();
    if (need_extra_info) {
      extra_infos = result.value()->GetExtraInfos();
    }
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

int HnswIndexHandler::knn_search(const vsag::DatasetPtr &query, int64_t topk,
                                 const std::string &parameters,
                                 const float *&dist, const int64_t *&ids,
                                 int64_t &result_size, float valid_ratio,
                                 int index_type, FilterInterface *bitmap,
                                 bool reverse_filter, bool need_extra_info,
                                 const char *&extra_infos, void *&iter_ctx,
                                 bool is_last_search, void *allocator)
{
  int ret = OB_SUCCESS;
  if (OB_FAIL(finish_hipvs_build())) { return ret; }
  std::function<bool(int64_t)> filter = [bitmap, reverse_filter](int64_t id) -> bool {
    if (!reverse_filter) {
      return bitmap->test(id);
    } else {
      return !(bitmap->test(id));
    }
  };
  std::function<bool(const char *)> exinfo_filter = [bitmap, reverse_filter](const char *data) -> bool {
    if (!reverse_filter) {
      return bitmap->test(data);
    } else {
      return !(bitmap->test(data));
    }
  };

  std::shared_ptr<ObVasgFilter> vsag_filter = std::make_shared<ObVasgFilter>(valid_ratio, filter, exinfo_filter);
  vsag::Allocator *vsag_allocator = nullptr;
  if (allocator != nullptr) vsag_allocator = static_cast<vsag::Allocator *>(allocator);
  vsag::IteratorContext *input_iter = static_cast<vsag::IteratorContext *>(iter_ctx);
  vsag::SearchParam search_param(true, parameters,
                                 bitmap == nullptr ? nullptr : vsag_filter,
                                 vsag_allocator, input_iter, is_last_search);
  tl::expected<std::shared_ptr<vsag::Dataset>, vsag::Error> result = index_->KnnSearch(query, topk, search_param);
  if (result.has_value()) {
    iter_ctx = search_param.iter_ctx;
    result.value()->Owner(false);
    ids = result.value()->GetIds();
    dist = result.value()->GetDistances();
    result_size = result.value()->GetDim();
    if (need_extra_info) {
      extra_infos = result.value()->GetExtraInfos();
    }
  } else {
    ret = vsag_errcode2ob(result.error().type);
  }
  return ret;
}

void set_log_level(int32_t ob_level_num)
{
  static std::map<int32_t, int32_t> ob2vsag_log_level = {
      {0 /*ERROR*/, vsag::Logger::Level::kERR},
      {1 /*WARN*/, vsag::Logger::Level::kWARN},
      {2 /*INFO*/, vsag::Logger::Level::kINFO},
      {3 /*EDIAG*/, vsag::Logger::Level::kERR},
      {4 /*WDIAG*/, vsag::Logger::Level::kWARN},
      {5 /*TRACE*/, vsag::Logger::Level::kTRACE},
      {6 /*DEBUG*/, vsag::Logger::Level::kDEBUG},
  };
  vsag::Options::Instance().logger()->SetLevel(
      static_cast<vsag::Logger::Level>(ob2vsag_log_level[ob_level_num]));
}

bool is_init_ = vsag::init();
bool is_init()
{
    LOG_INFO("[OBVSAG] Init VsagLib]:");
    if (is_init_) {
        LOG_INFO("[OBVSAG] Init VsagLib success");
    } else {
        LOG_INFO("[OBVSAG] Init VsagLib fail");
    }
    return is_init_; 
}

void set_logger(void *logger_ptr)
{
  vsag::Options::Instance().set_logger(static_cast<vsag::Logger *>(logger_ptr));
  vsag::Logger::Level log_level = static_cast<vsag::Logger::Level>(1); // default is debug level
  vsag::Options::Instance().logger()->SetLevel(log_level);
}

void set_block_size_limit(uint64_t size)
{
  vsag::Options::Instance().set_block_size_limit(size);
}

bool get_is_hgraph_type(uint8_t create_type) 
{
  bool res = false;
  switch (create_type) {
    case HNSW_TYPE: {
      res = false;
      break;
    }
    case HNSW_SQ_TYPE: 
    case HNSW_BQ_TYPE:
    case HGRAPH_TYPE: {
      res = true;
      break;
    }
  }
  return res;
}

const char* get_index_type_str(uint8_t create_type)
{
  const char* res;
  switch (create_type) {
    case HNSW_TYPE: {
      res = "hnsw";
      break;
    }
    case HNSW_SQ_TYPE: 
    case HNSW_BQ_TYPE:
    case HGRAPH_TYPE: {
      res = "hgraph";
      break;
    }
    case IPIVF_TYPE: {
      res = "sindi";
      break;
    }
  }
  return res;
}

const char* get_precise_quantization_type(const uint8_t type)
{
  const char* res = nullptr;
  if (type == QuantizationType::SQ8) {
    res = "sq8";
  } else {
    res = "fp32";
  }
  return res;
}


/**
  eg:
    hnsw: {
            "dtype": dtype, "metric_type": metric, "dim": dim, 
            "hnsw": {
              "max_degree": max_degree, "ef_construction": ef_construction, "ef_search": ef_search, "use_static": use_static
            }
          }
    hgraph: {
              "dtype": dtype, "metric_type": metric, "dim": dim, "extra_info_size": extra_info_size,
              "index_param": {
                "base_quantization_type": "fp32", "max_degree": max_degree, "ef_construction": ef_construction, "build_thread_count": 0
              }
            }
    sq: {
          "dtype": dtype, "metric_type": metric, "dim": dim, "extra_info_size": extra_info_size,
          "index_param": {
            "base_quantization_type": "sq8", "max_degree": max_degree, "ef_construction": ef_construction, "build_thread_count": 0
          }
        }
    bq: {
          "dtype": dtype, "metric_type": metric, "dim": dim, "extra_info_size": extra_info_size,
          "index_param": {
            "base_quantization_type": "rabitq", "max_degree": max_degree, "ef_construction": ef_construction, "build_thread_count": 0,
            "use_reorder": true, "ignore_reorder": true, "precise_quantization_type": "fp32", "precise_io_type": "block_memory_io"
          }
        }
*/
int construct_vsag_create_param(
    uint8_t create_type, const char *dtype, const char *metric, int dim,
    int max_degree, int ef_construction, int ef_search, void *allocator,
    int extra_info_size, int16_t refine_type, int16_t bq_bits_query,
    bool bq_use_fht, char *result_param_str)
{
  int ret = OB_SUCCESS;
  bool is_hgraph_type = get_is_hgraph_type(create_type);
  const char *index_type_str = is_hgraph_type ? "index_param" : "hnsw";
  const char *base_quantization_type;
  const int64_t buf_len = 1024;
  switch (create_type) {
  case HNSW_SQ_TYPE: {
    base_quantization_type = "sq8";
    break;
  }
  case HNSW_BQ_TYPE: {
    base_quantization_type = "rabitq";
    break;
  }
  case HGRAPH_TYPE: {
    base_quantization_type = "fp32";
    break;
  }
  default: {
    break;
  }
  }
  // ObIStreamBuf only supports seeking within the current callback buffer, while
  // VSAG's new format seeks to the footer. Keep the legacy format until global seek is supported.
  int64_t pos = 0;
  int64_t buff_size = 0;
  if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, "{\"dim\":%d",
                              int(dim)))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"dtype\":\"%s\"",
                                     dtype))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"metric_type\":\"%s\"",
                                     metric))) {
  } else if (extra_info_size > 0 &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                 ",\"extra_info_size\": %d",
                                 extra_info_size))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                 ",\"use_old_serial_format\":true"))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"%s\":{",
                                     index_type_str))) {
  } else if (OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos, "\"ef_construction\":%d",
                 ef_construction))) {
  } else if (! is_hgraph_type && OB_FAIL(databuff_printf(result_param_str,
                                 buf_len, pos, ",\"ef_search\":%d",
                                 ef_search))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"max_degree\":%d",
                                     max_degree))) {
  } else if (is_hgraph_type &&
      OB_FAIL(databuff_printf(
          result_param_str, buf_len, pos,
          ",\"base_quantization_type\":\"%s\"",
          base_quantization_type))) {
  } else if (is_hgraph_type &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"build_thread_count\":%d",
                                     0))) {
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos,
                 ",\"use_reorder\":true"))) {
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos,
                 ",\"ignore_reorder\":true"))) {
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos,
                 ",\"precise_quantization_type\":\"%s\"", get_precise_quantization_type(refine_type)))) {
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"precise_io_type\":\"block_memory_io\""))) {
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"rabitq_bits_per_dim_query\":%d", bq_bits_query))) {
  } else if (create_type == HNSW_BQ_TYPE &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     ",\"rabitq_use_fht\":%s", (bq_use_fht ? "true" : "false")))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                     "}}"))) {
  }
  if (OB_SUCC(ret)) {
    LOG_INFO("build param", K(create_type), KCSTRING(result_param_str), K(lbt()));
  }
  return ret;
}

int construct_vsag_sindi_create_param(uint8_t create_type, const char *dtype, const char *metric, 
    void *allocator, int extra_info_size, bool use_reorder, float doc_prune_ratio, int window_size,
    char *result_param_str)
{
  int ret = OB_SUCCESS;
  const char *index_type_str = "index_param";
  const int64_t buf_len = 1024;

  int64_t pos = 0;
  int64_t buff_size = 0;
  // ObIStreamBuf exposes the serialized index through callback-backed chunks.
  // Skip seek-based footer handling and let SINDI read from that stream directly;
  // BufferStreamReader otherwise treats the current chunk length as the full stream.
  const bool deserialize_without_footer = true;
  const bool deserialize_without_buffer = true;
  if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, "{\"dtype\":\"%s\"", dtype))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"metric_type\":\"%s\"", metric))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"dim\": 1024"))) {
  } else if (extra_info_size > 0 &&
             OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"extra_info_size\": %d", extra_info_size))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"%s\":{", index_type_str))) {
  } else if (OB_FAIL(databuff_printf(
                 result_param_str, buf_len, pos, "\"use_reorder\":%s", use_reorder ? "true" : "false"))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"doc_prune_ratio\":%f", doc_prune_ratio))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, ",\"window_size\":%d", window_size))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                 ",\"deserialize_without_footer\":%s",
                                 (deserialize_without_footer ? "true": "false")))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos,
                                 ",\"deserialize_without_buffer\":%s",
                                 (deserialize_without_buffer ? "true": "false")))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, buf_len, pos, "}}"))) {
  }
  if (OB_SUCC(ret)) {
    LOG_INFO("build param", K(create_type), KCSTRING(result_param_str), K(lbt()));
  }
  return ret;
}

/**
  eg:
    hnsw : {"hnsw": {"ef_search": ef_search, "skip_ratio": 0.7}}
    hgraph : {"hgraph": {"ef_search": ef_search, "use_extra_info_filter": use_extra_info_filter}}
*/
int construct_vsag_search_param(uint8_t create_type, 
                                int64_t ef_search, 
                                bool use_extra_info_filter, 
                                char *result_param_str)
{
  int ret = OB_SUCCESS;
  bool is_hgraph_type = get_is_hgraph_type(create_type);
  const char *index_type_str = is_hgraph_type ? "hgraph" : "hnsw";
  int64_t pos = 0;
  int64_t buff_size = 0;
  int64_t buf_len = 1024;
  if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "{\"%s\":{", index_type_str))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "\"ef_search\":%d", int(ef_search)))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        ",\"skip_ratio\":%f", 0.7))) {
  } else if (is_hgraph_type && OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        ",\"use_extra_info_filter\":%s", use_extra_info_filter ? "true" : "false"))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "}}"))) {
  }
  if (OB_SUCC(ret)) {
    LOG_TRACE("search param", KCSTRING(result_param_str), K(lbt()));
  }
  return ret;
}

int construct_vsag_sindi_search_param(float query_prune_ratio, uint64_t n_candidate, 
                                char *result_param_str)
{
  int ret = OB_SUCCESS;
  const char *index_type_str = "sindi";
  int64_t pos = 0;
  int64_t buff_size = 0;
  int64_t buf_len = 1024;
  if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "{\"%s\":{", index_type_str))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        "\"query_prune_ratio\":%f", query_prune_ratio))) {
  } else if (OB_FAIL(databuff_printf(result_param_str, 
                        buf_len, 
                        pos, 
                        ",\"n_candidate\":%lu}}", n_candidate))) {
  }
  if (OB_SUCC(ret)) {
    LOG_TRACE("search param", KCSTRING(result_param_str), K(lbt()));
  }
  return ret;
}

int create_index(VectorIndexPtr &index_handler,
                 IndexType index_type, const char *dtype,
                 const char *metric, int dim, int max_degree,
                 int ef_construction, int ef_search, void *allocator,
                 int extra_info_size /* = 0*/, int16_t refine_type /*= 0*/,
                 int16_t bq_bits_query /*= 32*/, bool bq_use_fht /*= false*/)
{
  int ret = OB_SUCCESS;
  if (dtype == nullptr || metric == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    vsag::Allocator *vsag_allocator = nullptr;
    if (allocator == nullptr) {
      vsag_allocator = nullptr;
      LOG_INFO("[OBVSAG] allocator is null , use default_allocator", K(index_type), K(lbt()));
    } else {
      vsag_allocator = static_cast<vsag::Allocator *>(allocator);
      LOG_INFO("[OBVSAG] use caller allocator ", K(index_type), K(lbt()));
    }
  
    adjust_create_index_max_degree(index_type, max_degree);

    const char* index_type_str = get_index_type_str(index_type);
    char result_param_str[1024] = {0};
    if (OB_FAIL(construct_vsag_create_param(
        uint8_t(index_type), dtype, metric, dim, max_degree, 
        ef_construction, ef_search, allocator, extra_info_size,
        refine_type, bq_bits_query, bq_use_fht, result_param_str))) {
    } else {
      const std::string input_json_str(result_param_str);
      tl::expected<std::shared_ptr<Index>, Error> index = vsag::Factory::CreateIndex(index_type_str, input_json_str, vsag_allocator);
      if (index.has_value()) {
        std::shared_ptr<vsag::Index> hnsw;
        hnsw = index.value();
        HnswIndexHandler *hnsw_index = new HnswIndexHandler(
            true, false, false, dtype, metric, max_degree, ef_construction,
            ef_search, dim, index_type, hnsw, vsag_allocator, extra_info_size,
            refine_type, bq_bits_query, bq_use_fht);
        if (OB_ISNULL(hnsw_index)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          index_handler = static_cast<VectorIndexPtr>(hnsw_index);
        }
      } else {
        ret = vsag_errcode2ob(index.error().type);
        LOG_WARN("[OBVSAG] create index error happend",
            K(ret), KCSTRING(result_param_str), K(index.error().type), KCSTRING(index.error().message.c_str()));
      }
    }
  }
  return ret;
}

int validate_create_index(const CreateIndexParam &param, std::string &err_msg)
{
  int ret = OB_SUCCESS;
  err_msg.clear();
  if (param.dtype_ == nullptr || param.metric_ == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    vsag::Allocator *vsag_allocator = nullptr;
    if (param.allocator_ == nullptr) {
      vsag_allocator = nullptr;
      LOG_INFO("[OBVSAG] allocator is null , use default_allocator", K(param.index_type_), K(lbt()));
    } else {
      vsag_allocator = static_cast<vsag::Allocator *>(param.allocator_);
      LOG_INFO("[OBVSAG] use caller allocator ", K(param.index_type_), K(lbt()));
    }

    const char *index_type_str = get_index_type_str(param.index_type_);
    char result_param_str[1024] = {0};
    if (param.is_sparse_) {
      if (OB_FAIL(construct_vsag_sindi_create_param(uint8_t(param.index_type_),
                                                    param.dtype_,
                                                    param.metric_,
                                                    param.allocator_,
                                                    param.extra_info_size_,
                                                    param.use_reorder_,
                                                    param.doc_prune_ratio_,
                                                    param.window_size_,
                                                    result_param_str))) {
      }
    } else {
      int max_degree = param.max_degree_;
      adjust_create_index_max_degree(param.index_type_, max_degree);
      if (OB_FAIL(construct_vsag_create_param(
          uint8_t(param.index_type_), param.dtype_, param.metric_, param.dim_, max_degree,
          param.ef_construction_, param.ef_search_, param.allocator_, param.extra_info_size_,
          param.refine_type_, param.bq_bits_query_, param.bq_use_fht_, result_param_str))) {
      }
    }
    if (OB_SUCC(ret)) {
      const std::string input_json_str(result_param_str);
      tl::expected<std::shared_ptr<Index>, Error> index =
          vsag::Factory::CreateIndex(index_type_str, input_json_str, vsag_allocator);
      if (!index.has_value()) {
        ret = vsag_errcode2ob(index.error().type);
        fill_vsag_error_message(index.error(), err_msg);
        LOG_WARN("[OBVSAG] validate create index error",
            K(ret), KCSTRING(result_param_str), K(index.error().type), KCSTRING(index.error().message.c_str()));
      }
    }
  }
  return ret;
}

int create_index(VectorIndexPtr &index_handler, IndexType index_type, const char *dtype, const char *metric,
    bool use_reorder, float doc_prune_ratio, int window_size, void *allocator, int extra_info_size /* = 0*/)
{
  int ret = OB_SUCCESS;
  if (dtype == nullptr || metric == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    vsag::Allocator *vsag_allocator = nullptr;
    if (allocator == nullptr) {
      vsag_allocator = nullptr;
      LOG_INFO("[OBVSAG] allocator is null , use default_allocator", K(index_type), K(lbt()));
    } else {
      vsag_allocator = static_cast<vsag::Allocator *>(allocator);
      LOG_INFO("[OBVSAG] use caller allocator ", K(index_type), K(lbt()));
    }

    const char *index_type_str = get_index_type_str(index_type);
    char result_param_str[1024] = {0};
    if (OB_FAIL(construct_vsag_sindi_create_param(uint8_t(index_type),
            dtype,
            metric,
            allocator,
            extra_info_size,
            use_reorder,
            doc_prune_ratio,
            window_size,
            result_param_str))) {
    } else {
      const std::string input_json_str(result_param_str);
      tl::expected<std::shared_ptr<Index>, Error> index =
          vsag::Factory::CreateIndex(index_type_str, input_json_str, vsag_allocator);
      if (index.has_value()) {
        std::shared_ptr<vsag::Index> hnsw;
        hnsw = index.value();
        HnswIndexHandler *hnsw_index = new HnswIndexHandler(true,
            false,
            false,
            dtype,
            metric,
            index_type,
            hnsw,
            vsag_allocator,
            extra_info_size,
            use_reorder,
            doc_prune_ratio,
            window_size);
        if (OB_ISNULL(hnsw_index)) {
          ret = OB_ALLOCATE_MEMORY_FAILED;
        } else {
          index_handler = static_cast<VectorIndexPtr>(hnsw_index);
        }
      } else {
        ret = vsag_errcode2ob(index.error().type);
        LOG_WARN("[OBVSAG] create index error happend", K(ret), KCSTRING(result_param_str), K(index.error().type));
      }
    }
  }
  return ret;
}

int build_index(VectorIndexPtr &index_handler, float *vector_list,
                int64_t *ids, int dim, int size, char *extra_infos /* = nullptr*/)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || vector_list == nullptr || ids == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    DatasetPtr dataset = vsag::Dataset::Make();
    dataset->Dim(dim)
        ->NumElements(size)
        ->Ids(ids)
        ->Float32Vectors(vector_list)
        ->Owner(false);
    if (extra_infos != nullptr) {
      dataset->ExtraInfos(extra_infos);
    }
    if (OB_FAIL(hnsw->build_index(dataset))) {
    }
  }
  return ret;
}

int build_index(VectorIndexPtr &index_handler, uint32_t *lens, uint32_t *dims, float *vals, int64_t *ids, int size,
    char *extra_info /* = nullptr*/)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || lens == nullptr || dims == nullptr || vals == nullptr || ids == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    uint32_t *cur_dims_ptr = dims;
    float *cur_vals_ptr = vals;
    std::vector<vsag::SparseVector> sparse_vectors(size);
    for (int i = 0; i < size; i++) {
      sparse_vectors[i].len_ = lens[i];
      sparse_vectors[i].ids_ = cur_dims_ptr;
      sparse_vectors[i].vals_ = cur_vals_ptr;
      cur_dims_ptr += lens[i];
      cur_vals_ptr += lens[i];
    }
    HnswIndexHandler *handler = static_cast<HnswIndexHandler *>(index_handler);
    DatasetPtr dataset = vsag::Dataset::Make();
    dataset->NumElements(size)->Ids(ids)->SparseVectors(sparse_vectors.data())->Owner(false);
    if (extra_info != nullptr) {
      dataset->ExtraInfos(extra_info);
    }
    if (OB_FAIL(handler->build_index(dataset))) {
    }
  }
  return ret;
}

int add_index(VectorIndexPtr &index_handler, float *vector,
              int64_t *ids, int dim, int size,
              char *extra_info /* = nullptr*/)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || vector == nullptr || ids == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    // add index
    DatasetPtr incremental = vsag::Dataset::Make();
    incremental->Dim(dim)
        ->NumElements(size)
        ->Ids(ids)
        ->Float32Vectors(vector)
        ->Owner(false);
    if (extra_info != nullptr) {
      incremental->ExtraInfos(extra_info);
    }
    if (OB_FAIL(hnsw->add_index(incremental))) {
    }
  }
  return ret;
}

int add_index(VectorIndexPtr &index_handler, uint32_t *lens, uint32_t *dims, float *vals, int64_t *ids, int size,
    char *extra_info /* = nullptr*/)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || lens == nullptr || dims == nullptr || vals == nullptr || ids == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    uint32_t *cur_dims_ptr = dims;
    float *cur_vals_ptr = vals;
    std::vector<vsag::SparseVector> sparse_vectors(size);
    for (int i = 0; i < size; i++) {
      sparse_vectors[i].len_ = lens[i];
      sparse_vectors[i].ids_ = cur_dims_ptr;
      sparse_vectors[i].vals_ = cur_vals_ptr;
      cur_dims_ptr += lens[i];
      cur_vals_ptr += lens[i];
    }
    const uint32_t MAX_DIM_LIMIT = 500000;
    uint32_t max_dim = 0;
    for (int i = 0; i < size && OB_SUCC(ret); i++) {
      uint32_t length = sparse_vectors[i].len_;
      for (int j = 0; j < length && OB_SUCC(ret); j++) {
        max_dim = MAX(max_dim, sparse_vectors[i].ids_[j]);
        if (OB_UNLIKELY(max_dim > MAX_DIM_LIMIT)) {
          ret = OB_NOT_SUPPORTED;
          LOG_USER_ERROR(OB_NOT_SUPPORTED, "sparse vector dimension greater than 500000 is");
        }
      }
    }
    if (OB_FAIL(ret)) {
      // do nothing
    } else {
      HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
      // add index
      DatasetPtr incremental = vsag::Dataset::Make();
      incremental->NumElements(size)->Ids(ids)->SparseVectors(sparse_vectors.data())->Owner(false);
      if (extra_info != nullptr) {
        incremental->ExtraInfos(extra_info);
      }
      if (OB_FAIL(hnsw->add_index(incremental))) {
      }
    }
  }
  return ret;
}

int get_index_type(VectorIndexPtr &index_handler)
{
  HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
  return hnsw->get_index_type();
}

int get_index_number(VectorIndexPtr &index_handler, int64_t &size)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    size = hnsw->get_index_number();
  }
  return ret;
}

int cal_distance_by_id(VectorIndexPtr &index_handler,
                       const float *vector, const int64_t *ids, int64_t count,
                       const float *&distances)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    if (OB_FAIL(hnsw->cal_distance_by_id(vector, ids, count, distances))) {
    }
  }
  return ret;
}

int cal_distance_by_id(VectorIndexPtr &index_handler, uint32_t len, uint32_t *dims, float *vals, const int64_t *ids,
    int64_t count, const float *&distances)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    if (OB_FAIL(hnsw->cal_distance_by_id(len, dims, vals, ids, count, distances))) {
    }
  }
  return ret;
}

int get_vid_bound(VectorIndexPtr &index_handler,
                         int64_t &min_vid, int64_t &max_vid)
{
  int ret = OB_SUCCESS;
  if (nullptr == index_handler) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    const IndexType index_type = static_cast<IndexType>(hnsw->get_index_type());
    if (index_type == IPIVF_TYPE) {
      // TODO(ningxin.ning): support get_vid_bound for ipivf
      min_vid = 0;
      max_vid = 0;
    } else {
      if (OB_FAIL(hnsw->get_vid_bound(min_vid, max_vid))) {
      }
    }
  }
  return ret;
}

int knn_search(VectorIndexPtr &index_handler, float *query_vector,
               int dim, int64_t topk, const float *&dist, const int64_t *&ids,
               int64_t &result_size, int ef_search, bool need_extra_info,
               const char *&extra_infos, void *invalid, bool reverse_filter,
               bool use_extra_info_filter, void *allocator, float valid_ratio, 
               float distance_threshold)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || query_vector == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    FilterInterface *bitmap = static_cast<FilterInterface *>(invalid);
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    const IndexType index_type = static_cast<IndexType>(hnsw->get_index_type());
    char result_param_str[1024]= {0};
    const int64_t EF_SEARCH_LIMIT = 1000L;
    const int64_t AMPLIFICATION_FACTOR = 10;
    if (ef_search > EF_SEARCH_LIMIT) {
      int64_t index_number = hnsw->get_index_number();
      if (0 != index_number) {
        topk = topk < index_number ? topk : index_number;
      }
      int64_t ef_search_threshold = AMPLIFICATION_FACTOR * topk > EF_SEARCH_LIMIT ? AMPLIFICATION_FACTOR * topk : EF_SEARCH_LIMIT;
      ef_search = ef_search < ef_search_threshold ? ef_search : ef_search_threshold;
    }
    if (OB_FAIL(construct_vsag_search_param(uint8_t(index_type), ef_search, use_extra_info_filter, result_param_str))) {
    } else {
      const std::string input_json_string(result_param_str);
      DatasetPtr query = vsag::Dataset::Make();
      query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);
      if (OB_FAIL(hnsw->knn_search(query, topk, input_json_string, dist, ids,
                                   result_size, valid_ratio, index_type, bitmap,
                                   reverse_filter, need_extra_info, extra_infos, allocator, distance_threshold))) {
      }
    }
  }
  return ret;
}

int knn_search(VectorIndexPtr &index_handler, float *query_vector,
               int dim, int64_t topk, const float *&dist, const int64_t *&ids,
               int64_t &result_size, int ef_search, bool need_extra_info,
               const char *&extra_infos, void *invalid, bool reverse_filter,
               bool use_extra_info_filter, float valid_ratio, void *&iter_ctx,
               bool is_last_search, void *allocator)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || query_vector == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    FilterInterface *bitmap = static_cast<FilterInterface *>(invalid);
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    const IndexType index_type = static_cast<IndexType>(hnsw->get_index_type());
    char result_param_str[1024]= {0};
    const int64_t EF_SEARCH_LIMIT = 1000L;
    const int64_t AMPLIFICATION_FACTOR = 10;
    if (ef_search > EF_SEARCH_LIMIT) {
      int64_t index_number = hnsw->get_index_number();
      if (0 != index_number) {
        topk = topk < index_number ? topk : index_number;
      }
      int64_t ef_search_threshold = AMPLIFICATION_FACTOR * topk > EF_SEARCH_LIMIT ? AMPLIFICATION_FACTOR * topk : EF_SEARCH_LIMIT;
      ef_search = ef_search < ef_search_threshold ? ef_search : ef_search_threshold;
    }
    if (OB_FAIL(construct_vsag_search_param(uint8_t(index_type), ef_search, use_extra_info_filter, result_param_str))) {
    } else {
      const std::string input_json_string(result_param_str);
      DatasetPtr query = vsag::Dataset::Make();
      query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);
      if (OB_FAIL(hnsw->knn_search(query, topk, input_json_string, dist, ids,
                            result_size, valid_ratio, index_type, bitmap,
                            reverse_filter, need_extra_info, extra_infos, iter_ctx,
                            is_last_search, allocator))) {
      }
    }
  }
  return ret;
}

int knn_search(obvsag::VectorIndexPtr &index_handler, uint32_t len, uint32_t *dims, float *vals, int64_t topk,
    const float *&result_dist, const int64_t *&result_ids, const char *&extra_infos, int64_t &result_size,
    float query_prune_ratio, int64_t n_candidate, void *invalid, bool reverse_filter,
    bool is_extra_info_filter, float valid_ratio, void *allocator, bool need_extra_info)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr || dims == nullptr || vals == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    FilterInterface *bitmap = static_cast<FilterInterface *>(invalid);
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    const IndexType index_type = static_cast<IndexType>(hnsw->get_index_type());
    char result_param_str[1024]= {0};
    if (OB_FAIL(construct_vsag_sindi_search_param(query_prune_ratio, n_candidate, result_param_str))) {
    } else if (len == 0) {
      result_size = 0;
    } else {
      // VSAG SINDI requires n_candidate <= 500 * k. SeekDB supports refine_k up to 1000,
      // so use the minimum internal k accepted by VSAG and let the caller merge and trim
      // the extra results to the SQL query limit. This preserves the requested candidate pool.
      const int64_t min_vsag_topk = n_candidate / VSAG_SINDI_N_CANDIDATE_FACTOR
                                    + (n_candidate % VSAG_SINDI_N_CANDIDATE_FACTOR != 0);
      const int64_t vsag_topk = std::max(topk, min_vsag_topk);
      const std::string input_json_string(result_param_str);
      vsag::SparseVector sparse;
      sparse.len_ = len;
      sparse.ids_ = dims;
      sparse.vals_ = vals;
      DatasetPtr query = vsag::Dataset::Make();
      query->NumElements(1)->SparseVectors(&sparse)->Owner(false);
      if (OB_FAIL(hnsw->knn_search(query, vsag_topk, input_json_string, result_dist, result_ids,
                            result_size, valid_ratio, index_type, bitmap,
                            reverse_filter, need_extra_info, extra_infos, allocator))) {
      }
    }
  }
  return ret;
}

int fserialize(VectorIndexPtr &index_handler, std::ostream &out_stream)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    if (OB_FAIL(hnsw->finish_hipvs_build())) { return ret; }
    tl::expected<void, Error> bs = hnsw->get_index()->Serialize(out_stream);
    if (bs.has_value()) {
      LOG_INFO("[OBVSAG] serialize index success");
    } else {
      ret = vsag_errcode2ob(bs.error().type);
      LOG_WARN("[OBVSAG] fserialize error happend", K(ret), K(bs.error().type));
    }
  }
  return ret;
}

int fdeserialize(VectorIndexPtr &index_handler,
                 std::istream &in_stream)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    std::shared_ptr<vsag::Index> hnsw_index;
    bool use_static = hnsw->get_use_static();
    const char *metric = hnsw->get_metric();
    const char *dtype = hnsw->get_dtype();
    int max_degree = hnsw->get_max_degree();
    int ef_construction = hnsw->get_ef_construction();
    int ef_search = hnsw->get_ef_search();
    int dim = hnsw->get_dim();
    int index_type = hnsw->get_index_type();
    uint64_t extra_info_size = hnsw->get_extra_info_size();
    const char* index_type_str = get_index_type_str(index_type);
    int16_t refine_type = hnsw->get_refine_type();
    int16_t bq_bits_query = hnsw->get_bq_bits_query();
    bool bq_use_fht = hnsw->get_bq_use_fht();
    bool use_reorder = hnsw->get_use_reorder();
    float doc_prune_ratio = hnsw->get_doc_prune_ratio();
    int window_size = hnsw->get_window_size();

    char result_param_str[1024] = {0};
    if ((IndexType)index_type == IndexType::IPIVF_TYPE) {
      if (OB_FAIL(construct_vsag_sindi_create_param(uint8_t(index_type),
              dtype,
              metric,
              hnsw->get_allocator(),
              extra_info_size,
              use_reorder,
              doc_prune_ratio,
              window_size,
              result_param_str))) {
      }
    } else {
      if (OB_FAIL(construct_vsag_create_param(
        uint8_t(index_type), dtype, metric, dim, max_degree,
        ef_construction, ef_search, hnsw->get_allocator(),
        extra_info_size, refine_type, bq_bits_query, bq_use_fht, result_param_str))) {
      } 
    }
    if (OB_FAIL(ret)) {
    } else {
      const std::string input_json_str(result_param_str);
      tl::expected<std::shared_ptr<Index>, Error> index = vsag::Factory::CreateIndex(index_type_str, input_json_str, hnsw->get_allocator());
      if (index.has_value()) {
        hnsw_index = index.value();
        tl::expected<void, Error> bs = hnsw_index->Deserialize(in_stream);
        if (bs.has_value()) {
          hnsw->set_index(hnsw_index);
          LOG_INFO("[OBVSAG] fdeserialize success", KCSTRING(result_param_str));
        } else {
          ret = vsag_errcode2ob(bs.error().type);
          LOG_WARN("[OBVSAG] fdeserialize error", K(ret), K(bs.error().type));
        }
      } else {
        ret = vsag_errcode2ob(index.error().type);
        LOG_WARN("[OBVSAG] create index error", K(ret), K(index.error().type));
      }
    }
  }
  return ret;
}

int delete_index(VectorIndexPtr &index_handler)
{
  int ret = OB_SUCCESS;
  LOG_INFO("[OBVSAG] delete index ",
      KP((void *)static_cast<HnswIndexHandler *>(index_handler)->get_index().get()),
      K(static_cast<HnswIndexHandler *>(index_handler)->get_index().use_count()), K(lbt()));
  if (index_handler != nullptr) {
    delete static_cast<HnswIndexHandler *>(index_handler);
    index_handler = nullptr;
  }
  return ret;
}

void delete_iter_ctx(void *iter_ctx)
{
  LOG_TRACE("[OBVAG] delete_iter_ctx", KP(iter_ctx), K(lbt()));
  if (iter_ctx != nullptr) {
    delete static_cast<vsag::IteratorContext *>(iter_ctx);
    iter_ctx = nullptr;
  }
}

int get_extra_info_by_ids(VectorIndexPtr &index_handler,
                          const int64_t *ids, int64_t count,
                          char *extra_infos)
{
  int ret = OB_SUCCESS;
  if (index_handler == nullptr) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    if (OB_FAIL(hnsw->get_extra_info_by_ids(ids, count, extra_infos))) {
    }
  }
  return ret;
}

uint64_t estimate_memory(VectorIndexPtr &index_handler, const uint64_t row_count, const bool is_build) 
{
  uint64_t estimate_memory_size = 0;
  if (index_handler != nullptr) {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    estimate_memory_size = hnsw->estimate_memory(row_count, is_build);
  }
  return estimate_memory_size;
}

int immutable_optimize(VectorIndexPtr& index_handler)
{
  int ret = OB_SUCCESS;
  if (OB_ISNULL(index_handler)) {
    ret = OB_INVALID_ARGUMENT;
  } else {
    HnswIndexHandler *hnsw = static_cast<HnswIndexHandler *>(index_handler);
    if (OB_FAIL(hnsw->immutable_optimize())) {
    }
  }
  return ret;
}

} // namespace obvsag
} // namespace common
} // namespace oceanbase
