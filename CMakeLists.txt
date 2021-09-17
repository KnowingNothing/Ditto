cmake_minimum_required(VERSION 3.2)
project(ditto C CXX)
set(LIB_NAME ditto)

# include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories("include")

# initial variables
set(NAST_LINKER_LIBS "")
set(NAST_RUNTIME_LINKER_LIBS ${CMAKE_DL_LIBS})

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tvm.cmake)

# Generic compilation options
# if(MSVC)
#   add_definitions(-DWIN32_LEAN_AND_MEAN)
#   add_definitions(-D_CRT_SECURE_NO_WARNINGS)
#   add_definitions(-D_SCL_SECURE_NO_WARNINGS)
#   add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
#   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
#   if(USE_MSVC_MT)
#     foreach(flag_var
#         CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
#         CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
#       if(${flag_var} MATCHES "/MD")
#         string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
#       endif(${flag_var} MATCHES "/MD")
#     endforeach(flag_var)
#   endif()
# else(MSVC)
#   if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
#     message("Build in Debug mode")
#     set(CMAKE_C_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
#   else()
#     set(CMAKE_C_FLAGS "-O2 -Wall -fPIC ${CMAKE_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
#     if (HIDE_PRIVATE_SYMBOLS)
#       message(STATUS "Hide private symbols...")
#       set(CMAKE_C_FLAGS "-fvisibility=hidden ${CMAKE_C_FLAGS}")
#       set(CMAKE_CXX_FLAGS "-fvisibility=hidden ${CMAKE_CXX_FLAGS}")
#     endif(HIDE_PRIVATE_SYMBOLS)
#   endif ()
#   if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
#       CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
#     set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
#   endif()
# endif(MSVC)


# Source file
# file(GLOB XCOMPILER_SRCS
#     src/*.cc
#     )

# file(GLOB XCOMPILER_SRCS
#     src/graph/*.cc
# )

# file(GLOB XCOMPILER_INCLUDES
#     include/*.h
# )

# file(GLOB XCOMPILER_INCLUDES
#     include/graph/*.h
# )


# if(NOT MSVC)
#   include(CheckCXXCompilerFlag)
#     check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
#     message(STATUS "Build with c++11")
#     set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
# endif()

# add_library(${LIB_NAME} SHARED ${XCOMPILER_SRCS} ${XCOMPILER_INCLUDES})
# target_compile_definitions(${LIB_NAME} PUBLIC DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
# target_link_libraries(${LIB_NAME} ${NAST_LINKER_LIBS} ${NAST_RUNTIME_LINKER_LIBS})

# if (HIDE_PRIVATE_SYMBOLS AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#   set(HIDE_SYMBOLS_LINKER_FLAGS "-Wl,--exclude-libs,ALL")
#   # Note: 'target_link_options' with 'PRIVATE' keyword would be cleaner
#   # but it's not available until CMake 3.13. Switch to 'target_link_options'
#   # once minimum CMake version is bumped up to 3.13 or above.
#   target_link_libraries(${LIB_NAME} ${HIDE_SYMBOLS_LINKER_FLAGS})
# endif()



include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/googletest.cmake)


add_subdirectory(test)
# add_subdirectory(project1)

# # JSON targets.
# add_subdirectory("3rdparty/jsoncpp"
#     ${CMAKE_CURRENT_BINARY_DIR}/jsoncpp-build
#     EXCLUDE_FROM_ALL)

# target_link_libraries(${LIB_NAME} jsoncpp_lib)

# set(CMAKE_CURRENT_SOURCE_DIR_OLD ${CMAKE_CURRENT_SOURCE_DIR})
# # set(CMAKE_CURRENT_BINARY_DIR_OLD ${CMAKE_CURRENT_BINARY_DIR})
# set(CMAKE_CURRENT_LIST_DIR_OLD ${CMAKE_CURRENT_LIST_DIR})
# set(CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tvm)
# # set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/3rdparty/tvm/)
# set(CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/tvm)
# # TVM targets.
# add_subdirectory("3rdparty/tvm"
#     ${CMAKE_CURRENT_BINARY_DIR}/tvm-build
#     EXCLUDE_FROM_ALL)

# target_link_libraries(${LIB_NAME} tvm-build)
