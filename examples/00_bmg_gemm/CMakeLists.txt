# Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set(TEST_BATCHES --l=2)
set(TEST_LARGE "--l=513 --m=8 --n=16384 --k=512") # B matrix capacity > uint32_max
set(TEST_SMALL_SHAPE --m=4 --n=8 --k=8 --l=2)


cutlass_example_add_executable(
  00_bmg_gemm
  00_bmg_gemm.cpp
  TEST_COMMAND_OPTIONS
  TEST_BATCHES
  TEST_LARGE
  TEST_SMALL_SHAPE
)

cutlass_example_add_executable(
  00_bmg_gemm_reducescatter
  00_bmg_gemm_reducescatter.cpp
  TEST_COMMAND_OPTIONS
  TEST_BATCHES
  TEST_LARGE
  TEST_SMALL_SHAPE
)

cutlass_example_add_executable(
  00_bmg_allgather_gemm
  00_bmg_allgather_gemm.cpp
  TEST_COMMAND_OPTIONS
  TEST_BATCHES
  TEST_LARGE
  TEST_SMALL_SHAPE
)

cutlass_example_add_executable(
  00_bmg_debug
  00_bmg_debug.cpp
  TEST_COMMAND_OPTIONS
  TEST_BATCHES
  TEST_LARGE
  TEST_SMALL_SHAPE
)

find_path(IMPI_INCLUDE_DIR mpi.h HINTS ENV I_MPI_ROOT PATH_SUFFIXES include)
find_library(IMPI_LIBRARY NAMES mpi impi HINTS ENV I_MPI_ROOT PATH_SUFFIXES lib lib/release)

if (NOT IMPI_INCLUDE_DIR OR NOT IMPI_LIBRARY)
  message(FATAL_ERROR "00_bmg_gemm_reducescatter requires Intel MPI. Source oneAPI setvars.sh so I_MPI_ROOT is available.")
endif()

target_include_directories(
  00_bmg_gemm_reducescatter
  PRIVATE
  ${IMPI_INCLUDE_DIR}
)

target_include_directories(
  00_bmg_allgather_gemm
  PRIVATE
  ${IMPI_INCLUDE_DIR}
)

target_include_directories(
  00_bmg_debug
  PRIVATE
  ${IMPI_INCLUDE_DIR}
)
  target_include_directories(
  00_bmg_gemm_reducescatter
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../gemm_reducescatter
)

target_include_directories(
  00_bmg_allgather_gemm
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../gemm_reducescatter
)

target_include_directories(
  00_bmg_debug
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../gemm_reducescatter
)

  find_library(ZE_LIBRARY NAMES ze_loader
    HINTS ENV ONEAPI_ROOT ENV LD_LIBRARY_PATH
    PATH_SUFFIXES lib lib64)

  target_link_libraries(
    00_bmg_gemm_reducescatter
    PRIVATE
    ${IMPI_LIBRARY}
    $<$<BOOL:${ZE_LIBRARY}>:${ZE_LIBRARY}>
  )

target_link_libraries(
  00_bmg_allgather_gemm
  PRIVATE
  ${IMPI_LIBRARY}
  $<$<BOOL:${ZE_LIBRARY}>:${ZE_LIBRARY}>
)

target_link_libraries(
  00_bmg_debug
  PRIVATE
  ${IMPI_LIBRARY}
  $<$<BOOL:${ZE_LIBRARY}>:${ZE_LIBRARY}>
)

cutlass_example_add_executable(
  00_bmg_symm_test
  symm_test.cpp
)

cutlass_example_add_executable(
  pingpong_latency_d2d
  pingpong_latency_d2d.cpp
)

target_include_directories(
  00_bmg_symm_test
  PRIVATE
  ${IMPI_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../gemm_reducescatter
)

target_include_directories(
  pingpong_latency_d2d
  PRIVATE
  ${IMPI_INCLUDE_DIR}
)

target_link_libraries(
  00_bmg_symm_test
  PRIVATE
  ${IMPI_LIBRARY}
  $<$<BOOL:${ZE_LIBRARY}>:${ZE_LIBRARY}>
)

target_link_libraries(
  pingpong_latency_d2d
  PRIVATE
  ${IMPI_LIBRARY}
  $<$<BOOL:${ZE_LIBRARY}>:${ZE_LIBRARY}>
)

set(TEST_SMALL_SHAPE_PADDABLE --m=1 --n=1 --k=2 --l=2)
cutlass_example_add_executable(
  00_bmg_gemm_padded
  00_bmg_gemm_padded.cpp
  TEST_COMMAND_OPTIONS
  TEST_BATCHES
  TEST_SMALL_SHAPE_PADDABLE
)

cutlass_example_add_executable(
  00_bmg_gemm_with_sycl_queue
  00_bmg_gemm_with_sycl_queue.cpp
  TEST_COMMAND_OPTIONS
  TEST_BATCHES
)

cutlass_example_add_executable(
  test_local_reduction
  test_local_reduction.cpp
)

cutlass_example_add_executable(
  00_bmg_gemm_allreduce
  00_bmg_gemm_allreduce.cpp
)

cutlass_example_add_executable(
  00_bmg_allreduce_standalone
  00_bmg_allreduce_standalone.cpp
)

target_include_directories(
  00_bmg_gemm_allreduce
  PRIVATE
  ${IMPI_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../gemm_reducescatter
)

target_include_directories(
  00_bmg_allreduce_standalone
  PRIVATE
  ${IMPI_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../gemm_reducescatter
)

target_link_libraries(
  00_bmg_gemm_allreduce
  PRIVATE
  ${IMPI_LIBRARY}
  $<$<BOOL:${ZE_LIBRARY}>:${ZE_LIBRARY}>
)

target_link_libraries(
  00_bmg_allreduce_standalone
  PRIVATE
  ${IMPI_LIBRARY}
  $<$<BOOL:${ZE_LIBRARY}>:${ZE_LIBRARY}>
)
