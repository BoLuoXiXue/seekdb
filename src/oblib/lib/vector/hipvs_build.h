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
#pragma once

#include <stdint.h>

// Keep the GPU compiler, HIP headers and C++ dependencies out of the server ABI.
// The output is a full, uncompressed FP32 HNSW index in hnswlib's file format.
struct HipvsBuildStats
{
  double graph_seconds;
  double hierarchy_seconds;
  double serialize_seconds;
  int32_t device;
  int32_t hierarchy_levels;
  uint64_t rows;
};

struct HipvsBuildInput
{
  uint64_t magic;
  int64_t rows;
  int32_t dim;
  int32_t max_degree;
  int32_t ef_construction;
  int32_t reserved;
};
static constexpr uint64_t HIPVS_BUILD_INPUT_MAGIC = 0x5344424849505631ULL;

using HipvsBuildFunction = int (*)(const float *, const int64_t *, int64_t, int32_t,
    int32_t, int32_t, const char *, HipvsBuildStats *, char *, uint64_t);

extern "C" int seekdb_hipvs_build_v1(const float *vectors, const int64_t *ids,
    int64_t count, int32_t dim, int32_t max_degree, int32_t ef_construction,
    const char *output, HipvsBuildStats *stats, char *error, uint64_t error_capacity);
