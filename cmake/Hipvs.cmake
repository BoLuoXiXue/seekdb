set(HIPVS_ROOT "/opt/hipvs" CACHE PATH "hipVS installation prefix")
find_path(HIPVS_ROCM_INCLUDE hip/hip_runtime.h HINTS /opt/rocm/include)
# The bridge only calls the host C API. GPU kernels and architecture support
# belong to the installed hipVS library, so building it needs no hipcc or
# architecture-specific code generation.
set(_hipvs_available OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ARCHITECTURE STREQUAL "x86_64"
    AND NOT CMAKE_CROSSCOMPILING
    AND HIPVS_ROCM_INCLUDE
    AND EXISTS "${HIPVS_ROOT}/include/cuvs/neighbors/hnsw.h"
    AND EXISTS "${HIPVS_ROOT}/lib/libcuvs_c.so")
  set(_hipvs_available ON)
endif()
option(OB_BUILD_HIPVS "Build the AMD hipVS HNSW construction worker" ${_hipvs_available})
if(OB_BUILD_HIPVS)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT ARCHITECTURE STREQUAL "x86_64")
    message(FATAL_ERROR "The hipVS bridge currently requires Linux x86_64")
  endif()
  if(NOT EXISTS "${HIPVS_ROOT}/include/cuvs/neighbors/hnsw.h"
      OR NOT EXISTS "${HIPVS_ROOT}/lib/libcuvs_c.so" OR NOT HIPVS_ROCM_INCLUDE)
    message(FATAL_ERROR "hipVS C headers/library and ROCm runtime headers are required")
  endif()
  set(_hipvs_source "${CMAKE_SOURCE_DIR}/src/oblib/lib/vector")
  add_library(hipvs_bridge SHARED "${_hipvs_source}/hipvs_build_bridge.cpp")
  target_include_directories(hipvs_bridge PRIVATE
    "${HIPVS_ROOT}/include" "${HIPVS_ROOT}/include/cuvs" "${HIPVS_ROCM_INCLUDE}")
  target_compile_definitions(hipvs_bridge PRIVATE __HIP_PLATFORM_AMD__=1)
  target_link_libraries(hipvs_bridge PRIVATE "${HIPVS_ROOT}/lib/libcuvs_c.so")
  set_target_properties(hipvs_bridge PROPERTIES
    OUTPUT_NAME seekdb_hipvs_build
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "${HIPVS_ROOT}/lib"
    INSTALL_RPATH "${HIPVS_ROOT}/lib")
  add_executable(hipvs_build_worker "${_hipvs_source}/hipvs_build_worker.cpp")
  set_target_properties(hipvs_build_worker PROPERTIES
    OUTPUT_NAME seekdb_hipvs_build_worker
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "$ORIGIN;$ORIGIN/../lib;$ORIGIN/../libexec/seekdb/lib;${HIPVS_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;$ORIGIN/../libexec/seekdb/lib;${HIPVS_ROOT}/lib")
  target_link_libraries(hipvs_build_worker PRIVATE hipvs_bridge)
  # A build-tree server and an installed server discover the worker beside
  # their executable; neither depends on the caller's working directory.
  add_custom_target(hipvs_build_runtime
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "$<TARGET_FILE:hipvs_build_worker>" "${CMAKE_BINARY_DIR}/src/observer/seekdb_hipvs_build_worker"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "$<TARGET_FILE:hipvs_bridge>" "${CMAKE_BINARY_DIR}/src/observer/libseekdb_hipvs_build.so"
    DEPENDS hipvs_build_worker hipvs_bridge
    VERBATIM)
  add_dependencies(seekdb hipvs_build_runtime)
  install(TARGETS hipvs_build_worker RUNTIME DESTINATION usr/bin COMPONENT server)
  install(TARGETS hipvs_bridge LIBRARY DESTINATION usr/libexec/seekdb/lib COMPONENT server)
else()
  # Reconfiguring an existing GPU build as CPU-only must not leave a worker
  # beside seekdb that automatic runtime discovery could still execute.
  file(REMOVE "${CMAKE_BINARY_DIR}/src/observer/seekdb_hipvs_build_worker"
    "${CMAKE_BINARY_DIR}/src/observer/libseekdb_hipvs_build.so")
endif()
message(STATUS "AMD hipVS construction worker: ${OB_BUILD_HIPVS}")
