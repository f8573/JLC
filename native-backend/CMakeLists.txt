cmake_minimum_required(VERSION 3.20)
project(jlc_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(JLC_NATIVE_ENABLE_VENDOR_BLAS "Enable vendor BLAS integration when available" ON)
option(JLC_NATIVE_ENABLE_MARCH_NATIVE "Enable -march=native for GCC/Clang/MinGW release builds" ON)
option(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM "Build reproducibility-only GEMM consumers, layouts and shared-A" OFF)
option(JLC_NATIVE_ENABLE_EXPERIMENTAL_ASM "Build the opt-in handwritten kmajor6 GEMM consumers" OFF)
set(JLC_R4_GENERATED_SOURCE_DIR "" CACHE PATH
  "Optional directory of Java-emitted R4 .cpp sources to link into jlc_native")
set(JLC_NATIVE_VENDOR_BLAS "AUTO" CACHE STRING "Vendor BLAS preference: AUTO, NONE, OPENBLAS, MKL")
set_property(CACHE JLC_NATIVE_VENDOR_BLAS PROPERTY STRINGS AUTO NONE OPENBLAS MKL)

if(NOT JLC_JAVA_HOME AND DEFINED ENV{JAVA_HOME})
  set(JLC_JAVA_HOME "$ENV{JAVA_HOME}")
endif()

if(NOT JLC_JAVA_HOME)
  message(FATAL_ERROR "JLC_JAVA_HOME must point to a JDK installation")
endif()

set(JLC_JNI_INCLUDE "${JLC_JAVA_HOME}/include")
if(WIN32)
  set(JLC_JNI_PLATFORM_INCLUDE "${JLC_JAVA_HOME}/include/win32")
elseif(APPLE)
  set(JLC_JNI_PLATFORM_INCLUDE "${JLC_JAVA_HOME}/include/darwin")
else()
  set(JLC_JNI_PLATFORM_INCLUDE "${JLC_JAVA_HOME}/include/linux")
endif()

if(NOT EXISTS "${JLC_JNI_INCLUDE}/jni.h")
  message(FATAL_ERROR "jni.h not found under ${JLC_JNI_INCLUDE}")
endif()

add_library(jlc_native_kernel OBJECT
  src/main/cpp/kernels/gemm.cpp
  src/main/cpp/kernels/gemm_config.cpp
  src/main/cpp/kernels/gemm_diagnostics.cpp
  src/main/cpp/kernels/lapack_factorizations.cpp
  src/main/cpp/kernels/lapack_hessenberg.cpp
  src/main/cpp/kernels/bidiagonal.cpp
)

# R4 generated-kernel registry is compiled with strict floating-point flags.
# It is intentionally separate from the existing GEMM target, whose historical
# benchmark path retains its own optimization policy.
add_library(jlc_native_codegen OBJECT
  src/main/cpp/codegen/jlc_generated_kernel_registry.cpp
)
target_include_directories(jlc_native_codegen PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/codegen"
)
if(MSVC)
  target_compile_options(jlc_native_codegen PRIVATE /O2)
else()
  target_compile_options(jlc_native_codegen PRIVATE -O2 -fno-fast-math -ffp-contract=off)
endif()

if(JLC_R4_GENERATED_SOURCE_DIR)
  file(GLOB JLC_R4_GENERATED_SOURCES CONFIGURE_DEPENDS
    "${JLC_R4_GENERATED_SOURCE_DIR}/*.cpp")
  if(JLC_R4_GENERATED_SOURCES)
    target_sources(jlc_native_codegen PRIVATE ${JLC_R4_GENERATED_SOURCES})
    foreach(JLC_R4_SOURCE IN LISTS JLC_R4_GENERATED_SOURCES)
      if(MSVC)
        set_source_files_properties(${JLC_R4_SOURCE} PROPERTIES
          COMPILE_OPTIONS "/O2;/fp:strict;/arch:AVX2")
      else()
        set_source_files_properties(${JLC_R4_SOURCE} PROPERTIES
          COMPILE_OPTIONS "-O2;-fno-fast-math;-ffp-contract=off;-mavx2")
      endif()
    endforeach()
    target_include_directories(jlc_native_codegen PRIVATE
      "${JLC_R4_GENERATED_SOURCE_DIR}")
  endif()
endif()

# Assembly is a subset of the experimental implementation. Preserve the existing
# assembly-only opt-in command by enabling its required source group as well.
if(JLC_NATIVE_ENABLE_EXPERIMENTAL_ASM)
  set(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM ON)
endif()
if(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
  target_sources(jlc_native_kernel PRIVATE src/main/cpp/kernels/gemm_experimental.cpp)
  target_compile_definitions(jlc_native_kernel PRIVATE JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM=1)
endif()

# The handwritten consumers are reproducibility-only and are not part of the
# normal production build. They use the Linux x86-64 SysV ABI and GNU
# assembler syntax; other platforms retain the intrinsic selector fallback.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$"
   AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"
   AND JLC_NATIVE_ENABLE_EXPERIMENTAL_ASM)
  enable_language(ASM)
  target_sources(jlc_native_kernel PRIVATE src/main/cpp/kernels/kmajor6_6x8_u4.S
    src/main/cpp/kernels/kmajor6_6x8_u4_nr8.S)
  target_compile_definitions(jlc_native_kernel PRIVATE JLC_NATIVE_HAS_KMAJOR6_ASM=1)
endif()

target_include_directories(jlc_native_kernel PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/api"
)

add_library(jlc_native SHARED
  src/main/cpp/jni/jlc_native_Jni.cpp
  $<TARGET_OBJECTS:jlc_native_kernel>
  $<TARGET_OBJECTS:jlc_native_codegen>
)

target_include_directories(jlc_native PRIVATE
  "${JLC_JNI_INCLUDE}"
  "${JLC_JNI_PLATFORM_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/api"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/codegen"
)

add_executable(jlc_native_gemm_sweep
  src/main/cpp/bench/gemm_sweep.cpp
  $<TARGET_OBJECTS:jlc_native_kernel>
)

target_include_directories(jlc_native_gemm_sweep PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/api"
)

set(JLC_VENDOR_NAME "builtin")
if(JLC_NATIVE_ENABLE_VENDOR_BLAS AND NOT JLC_NATIVE_VENDOR_BLAS STREQUAL "NONE")
  set(JLC_NATIVE_VENDOR_EXPLICIT OFF)
  if(JLC_NATIVE_VENDOR_BLAS STREQUAL "OPENBLAS")
    set(BLA_VENDOR "OpenBLAS")
    set(JLC_NATIVE_VENDOR_EXPLICIT ON)
  elseif(JLC_NATIVE_VENDOR_BLAS STREQUAL "MKL")
    set(BLA_VENDOR "Intel10_64lp")
    set(JLC_NATIVE_VENDOR_EXPLICIT ON)
  endif()

  find_package(BLAS QUIET)
  find_path(JLC_CBLAS_INCLUDE_DIR NAMES cblas.h)
  if(BLAS_FOUND AND JLC_CBLAS_INCLUDE_DIR)
    target_link_libraries(jlc_native PRIVATE BLAS::BLAS)
    target_link_libraries(jlc_native_gemm_sweep PRIVATE BLAS::BLAS)
    target_include_directories(jlc_native_kernel PRIVATE "${JLC_CBLAS_INCLUDE_DIR}")
    target_include_directories(jlc_native PRIVATE "${JLC_CBLAS_INCLUDE_DIR}")
    target_include_directories(jlc_native_gemm_sweep PRIVATE "${JLC_CBLAS_INCLUDE_DIR}")
    if(BLA_VENDOR)
      set(JLC_VENDOR_NAME "${BLA_VENDOR}")
    else()
      set(JLC_VENDOR_NAME "cblas")
    endif()
    target_compile_definitions(jlc_native_kernel PRIVATE JLC_NATIVE_HAS_VENDOR_BLAS=1)

    find_package(LAPACK QUIET)
    if(LAPACK_FOUND)
      target_link_libraries(jlc_native PRIVATE LAPACK::LAPACK)
      target_link_libraries(jlc_native_gemm_sweep PRIVATE LAPACK::LAPACK)
      target_compile_definitions(jlc_native_kernel PRIVATE JLC_NATIVE_HAS_VENDOR_LAPACK=1)
    elseif(JLC_NATIVE_VENDOR_EXPLICIT)
      message(FATAL_ERROR "Requested ${JLC_NATIVE_VENDOR_BLAS} BLAS was found, but compatible LAPACK was not found")
    endif()
  elseif(JLC_NATIVE_VENDOR_EXPLICIT)
    message(FATAL_ERROR "Requested ${JLC_NATIVE_VENDOR_BLAS} vendor BLAS was not found. Set JLC_NATIVE_VENDOR_BLAS=AUTO or NONE to allow builtin fallback.")
  endif()
endif()

message(STATUS "JLC native vendor provider: ${JLC_VENDOR_NAME}")

target_compile_definitions(jlc_native_kernel PRIVATE
  JLC_NATIVE_PHASE=4
  JLC_NATIVE_VENDOR_BLAS_NAME="${JLC_VENDOR_NAME}"
)

function(jlc_native_apply_optimization target_name)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /O2 /arch:AVX2)
  else()
    target_compile_options(${target_name} PRIVATE -O3 -DNDEBUG -ffast-math)
    if(JLC_NATIVE_ENABLE_MARCH_NATIVE)
      target_compile_options(${target_name} PRIVATE -march=native)
    endif()
  endif()
endfunction()

jlc_native_apply_optimization(jlc_native_kernel)
jlc_native_apply_optimization(jlc_native)
jlc_native_apply_optimization(jlc_native_gemm_sweep)

if(MINGW)
  target_link_options(jlc_native PRIVATE -static-libgcc -static-libstdc++)
  target_link_options(jlc_native_gemm_sweep PRIVATE -static-libgcc -static-libstdc++)
endif()

set_target_properties(jlc_native PROPERTIES
  OUTPUT_NAME "jlc_native"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

set_target_properties(jlc_native_gemm_sweep PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

if(WIN32)
  set_target_properties(jlc_native PROPERTIES PREFIX "")
endif()

# Test hooks never enter a normal library build. Fault builds are validation-only.
option(JLC_NATIVE_ENABLE_TEST_HOOKS "Enable GEMM failure injection for native validation" OFF)
option(JLC_NATIVE_BUILD_TESTS "Build native GEMM contract tests" OFF)
if(JLC_NATIVE_ENABLE_TEST_HOOKS)
  target_compile_definitions(jlc_native_kernel PRIVATE
    JLC_NATIVE_TEST_POOL_FAULT_INJECTION=1 JLC_NATIVE_TEST_WORKER_FAULT_INJECTION=1)
  if(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    target_compile_definitions(jlc_native_kernel PRIVATE JLC_NATIVE_TEST_A_PACKING_FAULT_INJECTION=1)
  endif()
endif()
if(JLC_NATIVE_BUILD_TESTS)
  enable_testing()
  add_executable(gemm_contract_validation src/test/cpp/gemm_contract_validation.cpp)
  target_include_directories(gemm_contract_validation PRIVATE src/main/cpp/api)
  target_link_libraries(gemm_contract_validation PRIVATE jlc_native)
  if(JLC_NATIVE_ENABLE_EXPERIMENTAL_GEMM)
    add_test(NAME gemm_contract COMMAND gemm_contract_validation experiments)
  else()
    add_test(NAME gemm_contract COMMAND gemm_contract_validation)
  endif()
  if(JLC_NATIVE_ENABLE_TEST_HOOKS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(gemm_worker_failure_validation src/test/cpp/gemm_worker_failure_validation.cpp)
    target_include_directories(gemm_worker_failure_validation PRIVATE src/main/cpp/api)
    target_link_libraries(gemm_worker_failure_validation PRIVATE jlc_native)
    add_test(NAME gemm_worker_failure COMMAND gemm_worker_failure_validation)
    set_tests_properties(gemm_worker_failure PROPERTIES TIMEOUT 60)
  endif()
endif()
