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

#include "hipvs_build.h"
#include <vsag/allocator.h>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <new>
#include <vector>

namespace oceanbase {
namespace common {
namespace obvsag {

template <typename T>
class HipvsBufferAllocator
{
public:
  using value_type = T;
  explicit HipvsBufferAllocator(vsag::Allocator *allocator = nullptr) : allocator_(allocator) {}
  template <typename U>
  HipvsBufferAllocator(const HipvsBufferAllocator<U> &other) : allocator_(other.allocator_) {}
  T *allocate(size_t count)
  {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) { throw std::bad_alloc(); }
    void *data = allocator_ ? allocator_->Allocate(count * sizeof(T)) : std::malloc(count * sizeof(T));
    if (!data) { throw std::bad_alloc(); }
    return static_cast<T *>(data);
  }
  void deallocate(T *data, size_t)
  {
    if (allocator_) { allocator_->Deallocate(data); } else { std::free(data); }
  }
  template <typename U>
  bool operator==(const HipvsBufferAllocator<U> &other) const { return allocator_ == other.allocator_; }
  template <typename U>
  bool operator!=(const HipvsBufferAllocator<U> &other) const { return !(*this == other); }
  vsag::Allocator *allocator_;
};

struct HipvsBuildBuffer
{
  explicit HipvsBuildBuffer(vsag::Allocator *allocator)
      : vectors(HipvsBufferAllocator<float>(allocator)), ids(HipvsBufferAllocator<int64_t>(allocator)) {}
  void release()
  {
    decltype(vectors)(vectors.get_allocator()).swap(vectors);
    decltype(ids)(ids.get_allocator()).swap(ids);
    pending = false;
  }
  std::mutex mutex;
  bool pending = true;
  bool built_on_gpu = false;
  std::vector<float, HipvsBufferAllocator<float>> vectors;
  std::vector<int64_t, HipvsBufferAllocator<int64_t>> ids;
};
} // namespace obvsag
} // namespace common
} // namespace oceanbase
