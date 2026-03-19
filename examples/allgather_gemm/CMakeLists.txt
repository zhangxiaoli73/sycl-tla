# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (C) 2025 Intel Corporation, All rights reserved.
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

# set(CUTLASS_ISHEMM_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

function(add_cutlass_ishmem_executable NAME)

  set(options)
  set(oneValueArgs DISABLE_TESTS)
  set(multiValueArgs DEPENDS DEPENDEES TEST_COMMAND_OPTIONS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED __DISABLE_TESTS)
    set(__DISABLE_TESTS OFF)
  endif()

  cutlass_add_executable(${NAME} ${__UNPARSED_ARGUMENTS} BATCH_SOURCES OFF)

  add_dependencies(cutlass_examples ${NAME})

  # it will cause Sycl compilation error. use ISHMEM_PATH instead
  # find_package(ISHMEM REQUIRED)

  find_library(LEVELZERO_LIBRARY NAMES ze_loader)

  target_link_libraries(
    ${NAME}
    PRIVATE
    CUTLASS
    cutlass_tools_util_includes
    # ISHMEM::ISHMEM
    $ENV{ISHMEM_PATH}/lib/libishmem.a
    ${LEVELZERO_LIBRARY}
    )

  target_include_directories(
    ${NAME}
    PRIVATE
    $ENV{ISHMEM_PATH}/include/
    ${CUTLASS_EXAMPLES_UTILS_DIR}
    ${CUTLASS_APPLICATIONS_DIR}
    )

  if (CUTLASS_ENABLE_SYCL)
    add_onemkl_to_target(TARGET ${NAME})
    add_sycl_to_target(TARGET ${NAME})
  endif()

  install(
    TARGETS ${NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

  cutlass_add_executable_tests(
    test_examples_${NAME} ${NAME}
    DEPENDS ${__DEPENDS}
    DEPENDEES test_examples ${__DEPENDEES}
    TEST_COMMAND_OPTIONS ${__TEST_COMMAND_OPTIONS}
    DISABLE_EXECUTABLE_INSTALL_RULE
    DISABLE_TESTS ${__DISABLE_TESTS}
    )

endfunction()

message(STATUS "Building distributed-gemm for Intel GPU targets")

# set(CUTLASS_ISHEMM_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_cutlass_ishmem_executable(
  allgather_gemm_main
  allgather_gemm_main.cpp
  )

