# ggml-axcl: Axera AXCL NPU backend (AX8850 / AX650 M.2 accelerator cards)
#
# The AXCL SDK ships with the axclhost package (M5Stack apt repo) and installs
# headers to /usr/include/axcl and the runtime library into the axcl ld.so
# path. For development, point AXCL_ROOT at a directory containing an
# axcl-include/ (headers) and lib/ (runtime libs).

if (NOT UNIX)
    message(FATAL_ERROR "ggml-axcl: AXCL supports unix but not ${CMAKE_SYSTEM_NAME}")
endif()

if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64")
    # x86 hosts can compile the backend (e.g. syntax checks against vendored
    # headers) but linking libaxcl_rt requires an aarch64 target
else()
    message(FATAL_ERROR "ggml-axcl: unsupported processor ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(AXCL_INSTALL_DIR "" CACHE PATH "AXCL SDK root (contains include/ and lib/)")

if (AXCL_INSTALL_DIR)
    set(AXCL_INCLUDE_DIRS ${AXCL_INSTALL_DIR}/include)
    set(AXCL_LIB_DIRS ${AXCL_INSTALL_DIR}/lib)
else()
    # system install from the axclhost package
    set(AXCL_INCLUDE_DIRS /usr/include/axcl)
    set(AXCL_LIB_DIRS /usr/lib/axcl)
endif()

find_library(AXCL_RT_LIB axcl_rt HINTS ${AXCL_LIB_DIRS} REQUIRED)
# libaxcl_npu registers the NPU device provider; without it the runtime
# reports "device 0 is not connected" even with the driver loaded
find_library(AXCL_NPU_LIB axcl_npu HINTS ${AXCL_LIB_DIRS} REQUIRED)

file(GLOB GGML_SOURCES_AXCL "*.cpp")

ggml_add_backend_library(ggml-axcl ${GGML_SOURCES_AXCL})
# keep libaxcl_npu in the link even though we reference no symbols from it:
# its static initializers register the NPU device provider, and the linker's
# --as-needed default silently drops it otherwise
target_link_libraries(ggml-axcl PRIVATE
    -Wl,--no-as-needed
    ${AXCL_RT_LIB}
    ${AXCL_NPU_LIB}
    -Wl,--as-needed
)
target_include_directories(ggml-axcl PRIVATE ${AXCL_INCLUDE_DIRS})

message(STATUS "ggml-axcl: AXCL_INCLUDE_DIRS = ${AXCL_INCLUDE_DIRS}")
message(STATUS "ggml-axcl: AXCL_RT_LIB       = ${AXCL_RT_LIB}")
