cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0045 NEW)

project(bctppl-tools)
set(VERSION 0.7.0)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_MODULE_PATH ${ctas_SOURCE_DIR}/cmake)
set(CMAKE_C_STANDARD "17")
set(CMAKE_CXX_STANDARD "17")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -pthread")
set(CMAKE_CXX_FLAGS "$ENV{CPPFLAGS} ${CMAKE_CXX_FLAGS} -ffast-math -pthread")

# uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# dist target
if(NOT TARGET dist)
  add_custom_target(dist
    COMMAND git archive --prefix=${PROJECT_NAME}.${VERSION}/ HEAD | bzip2 >
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.${VERSION}.tar.bz2
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

FIND_PACKAGE (OpenCL REQUIRED)
include_directories("${OpenCL_INCLUDE_DIRS}")
add_definitions(-DOPENCL_FOUND)
add_definitions(-DCL_TARGET_OPENCL_VERSION=220)
set(OPENCL_LL "OpenCL")

#find_package(Torch REQUIRED)
#find_package(PythonExtensions REQUIRED)
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

find_library(POPTL poptmx)
find_path(POPTH poptmx.h REQUIRED)
include_directories(${POPTH})

find_library(CTASCL common PATH_SUFFIXES ctas REQUIRED)
find_path(CTASH ctas/common.world.h REQUIRED)
include_directories(${CTASH})

add_executable(align align.cpp)
target_link_libraries(align
  poptmx
  ${CTASCL}
)

#add_library(remove_borders_ext MODULE
#    remove_borders.cu
#)
#python_extension_module(remove_borders_ext)
#target_link_libraries(remove_borders_ext ${TORCH_LIBRARIES})
#target_include_directories(remove_borders_ext PRIVATE ${TORCH_INCLUDE_DIRS})
#set_property(TARGET remove_borders_ext PROPERTY CXX_STANDARD 14)

install(TARGETS align RUNTIME DESTINATION bin)



