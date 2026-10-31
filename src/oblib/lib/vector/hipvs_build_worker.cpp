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
#include <cstdio>
#include <fstream>
#include <limits>
#include <signal.h>
#include <stdexcept>
#include <sys/prctl.h>
#include <unistd.h>
#include <vector>

// The worker deliberately links neither VSAG nor seekdb's malloc interposer.
// No C++ objects or allocator-owned pointers cross the process boundary.
int main(int argc, char **argv)
{
  try {
    if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || getppid() == 1) {
      throw std::runtime_error("GPU worker lost its parent");
    }
    if (argc != 4) { throw std::invalid_argument("usage: worker INPUT OUTPUT STATS"); }
    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    const auto length = input.tellg();
    input.seekg(0);
    HipvsBuildInput header{};
    if (!input.read(reinterpret_cast<char *>(&header), sizeof(header))
        || header.magic != HIPVS_BUILD_INPUT_MAGIC || header.rows < 256
        || header.rows > std::numeric_limits<uint32_t>::max() || header.dim <= 0
        || header.dim > 65536) {
      throw std::invalid_argument("invalid GPU build input header");
    }
    const uint64_t floats = static_cast<uint64_t>(header.rows) * header.dim;
    const uint64_t expected = sizeof(header) + floats * sizeof(float) + header.rows * sizeof(int64_t);
    if (length < 0 || static_cast<uint64_t>(length) != expected) {
      throw std::invalid_argument("incomplete GPU build input");
    }
    std::vector<float> vectors(floats);
    std::vector<int64_t> ids(header.rows);
    if (!input.read(reinterpret_cast<char *>(vectors.data()), floats * sizeof(float))
        || !input.read(reinterpret_cast<char *>(ids.data()), ids.size() * sizeof(int64_t))) {
      throw std::runtime_error("cannot read GPU build vectors and IDs");
    }
    input.close();
    HipvsBuildStats stats{};
    char error[2048] = {};
    if (seekdb_hipvs_build_v1(vectors.data(), ids.data(), header.rows, header.dim,
        header.max_degree, header.ef_construction, argv[2], &stats, error, sizeof(error))) {
      throw std::runtime_error(error);
    }
    std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
    if (!output.write(reinterpret_cast<const char *>(&stats), sizeof(stats))) {
      throw std::runtime_error("cannot write GPU build statistics");
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "hipVS worker: %s\n", e.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr, "hipVS worker: unknown failure\n");
    return 1;
  }
}
