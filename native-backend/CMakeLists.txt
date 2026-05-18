cmake_minimum_required(VERSION 3.20)
project(jlc_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(JLC_NATIVE_ENABLE_VENDOR_BLAS "Enable vendor BLAS integration when available" ON)
option(JLC_NATIVE_ENABLE_MARCH_NATIVE "Enable -march=native for GCC/Clang/MinGW release builds" ON)
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
  src/main/cpp/kernels/lapack_factorizations.cpp
  src/main/cpp/kernels/lapack_hessenberg.cpp
  src/main/cpp/kernels/bidiagonal.cpp
)

target_include_directories(jlc_native_kernel PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/api"
)

add_library(jlc_native SHARED
  src/main/cpp/jni/jlc_native_Jni.cpp
  $<TARGET_OBJECTS:jlc_native_kernel>
)

target_include_directories(jlc_native PRIVATE
  "${JLC_JNI_INCLUDE}"
  "${JLC_JNI_PLATFORM_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/main/cpp/api"
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
