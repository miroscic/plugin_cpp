#   ____  _             _       
#  |  _ \| |_   _  __ _(_)_ __  
#  | |_) | | | | |/ _` | | '_ \ 
#  |  __/| | |_| | (_| | | | | |
#  |_|   |_|\__,_|\__, |_|_| |_|
#                 |___/         
cmake_minimum_required(VERSION 3.20)
project(plugin VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

# PROJECT SETTINGS #############################################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(USR_DIR ${CMAKE_CURRENT_LIST_DIR}/usr)
set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(CMAKE_INSTALL_PREFIX ${USR_DIR})
set(FETCHCONTENT_QUIET OFF)

# DEPENDENCIES #################################################################
if(WIN32)
  # On Windows, we rely on env variables to find install paths
  if(NOT DEFINED ENV{OpenCV_DIR})
    message(FATAL_ERROR "OpenCV_DIR env var not set, please set it to the OpenCV install directory")
  endif()
  set(OpenCV_DIR "$ENV{OpenCV_DIR}\\build" CACHE PATH "OpenCV build directory" FORCE)
  if(NOT DEFINED ENV{OpenVINO_DIR})
    message(FATAL_ERROR "OpenVINO_DIR env var not set, please set it to the OpenVINO install directory")
  endif()
  set(OpenVINO_DIR "$ENV{OpenVINO_DIR}\\runtime\\cmake" CACHE PATH "OpenVINO install directory" FORCE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core highgui videoio imgproc imgcodecs)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

if(WIN32)
  # OpenVINO headers have to be manually found on Windows
  set(OPENVINO_INCLUDE_DIR "$ENV{OpenVINO_DIR}\\runtime\\include")
else()
  find_path(OPENVINO_INCLUDE_DIR NAMES openvino.hpp 
    PATHS /opt/homebrew/include /usr/local/include
    PATH_SUFFIXES openvino
  )
  cmake_path(GET OPENVINO_INCLUDE_DIR PARENT_PATH OPENVINO_INCLUDE_DIR)
endif()
if (NOT OPENVINO_INCLUDE_DIR)
  message(FATAL_ERROR "OpenVINO not found")
else()
  message(STATUS "OpenVINO headers found at ${OPENVINO_INCLUDE_DIR}")
endif()

include(FetchContent)
# pugg is for the plugin system
FetchContent_Declare(pugg 
  GIT_REPOSITORY https://github.com/pbosetti/pugg.git
  GIT_TAG        1.0.2
  GIT_SHALLOW    TRUE
)
# plugins deal with I/O as JSON objects
set(BUILD_TESTING OFF CACHE INTERNAL "")
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pugg json)

# Compiler options #############################################################
if(MSVC)
  add_compile_options(/wd4267)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wno-deprecated-anon-enum-enum-conversion)
endif()

# MACROS #######################################################################
# Call: add_plugin(name [SRCS src1 src2 ...] [LIBS lib1 lib2 ...])
#       the source file ${SRC_DIR}/plugin/<name>.cpp is implicitly added
macro(add_plugin name)
  # on MacOS only, plugins can be compiled as executables
  set(multiValueArgs LIBS SRCS)
  cmake_parse_arguments(plugin "" "" "${multiValueArgs}" ${ARGN})
  if (APPLE)
    add_executable(${name} ${SRC_DIR}/plugin/${name}.cpp ${plugin_SRCS})
    set_target_properties(${name} PROPERTIES ENABLE_EXPORTS TRUE)
    set(${name}_EXEC ${name}.plugin)
  else()
    add_library(${name} SHARED ${SRC_DIR}/plugin/${name}.cpp ${plugin_SRCS})
    add_executable(${name}_main ${SRC_DIR}/plugin/${name}.cpp ${plugin_SRCS})
    target_link_libraries(${name}_main PRIVATE pugg ${plugin_LIBS})
    set_target_properties(${name}_main PROPERTIES OUTPUT_NAME ${name})
    set(${name}_EXEC ${name})
    list(APPEND TARGET_LIST ${name}_main)
  endif()
  target_link_libraries(${name} PRIVATE pugg ${plugin_LIBS})
  set_target_properties(${name} PROPERTIES PREFIX "")
  set_target_properties(${name} PROPERTIES SUFFIX ".plugin")
  target_compile_definitions(${name} PRIVATE PLUGIN_NAME="${name}")
  list(APPEND TARGET_LIST ${name})
endmacro()

macro(add_loader name)
  add_executable(${name} ${SRC_DIR}/main/${name}.cpp)
  target_link_libraries(${name} PRIVATE pugg)
  list(APPEND TARGET_LIST ${name})
endmacro()


# BUILD SETTINGS ###############################################################
if (APPLE)
  set(CMAKE_INSTALL_RPATH "@executable_path/../lib")
else()
  set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib;/usr/local/lib")
endif()
include_directories(${json_SOURCE_DIR}/include)

# OMZ common source files ######################################################
set(COMMON ${SRC_DIR}/common)

file(GLOB COMMON_SOURCES
  ${COMMON}/models/src/*.cpp
  ${COMMON}/monitors/src/*.cpp
  ${COMMON}/pipelines/src/*.cpp
  ${COMMON}/utils/src/*.cpp
)
file(GLOB COMMON_HEADERS
  ${COMMON}/models/include/models/*.h
  ${COMMON}/monitors/include/monitors/*.h
  ${COMMON}/pipelines/include/pipelines/*.h
  ${COMMON}/utils/include/utils/*.h*
)
# Remove items that depend on gflags (which we don't want)
list(REMOVE_ITEM COMMON_SOURCES ${COMMON}/monitors/src/query_wrapper.cpp)
list(REMOVE_ITEM COMMON_HEADERS ${COMMON}/monitors/src/query_wrapper.h)
list(REMOVE_ITEM COMMON_SOURCES ${COMMON}/utils/src/args_helper.cpp)
list(REMOVE_ITEM COMMON_HEADERS ${COMMON}/utils/include/utils/default_flags.hpp)
# locally modified files
list(APPEND COMMON_SOURCES ${SRC_DIR}/args_helper.cpp)
list(APPEND COMMON_SOURCES ${SRC_DIR}/args_helper.cpp)

# Add the include directories ##################################################
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${COMMON}/models/include
  ${COMMON}/monitors/include
  ${COMMON}/pipelines/include
  ${COMMON}/utils/include
  ${OPENVINO_INCLUDE_DIR}
  ${json_SOURCE_DIR}/include
)

# TARGETS ######################################################################

add_library(common STATIC ${COMMON_SOURCES} ${COMMON_HEADERS})

# add_plugin(echo)
# add_plugin(echoj)
# add_plugin(twice)
# add_plugin(clock)
# add_loader(loader)
# add_loader(load_filter)

add_loader(load_source)
#add_plugin(quick LIBS common ${OpenCV_LIBS} openvino::runtime)
add_plugin(skeletonizer3D LIBS common ${OpenCV_LIBS} openvino::runtime)

# Download OpenPOSE models #####################################################
set(MODEL_NAME "human-pose-estimation-0001" CACHE STRING "Model name (no extension)")
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/${MODEL_NAME}.xml)
  message(STATUS "Downloading OpenPOSE model")
  file(DOWNLOAD https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/${MODEL_NAME}/FP32/${MODEL_NAME}.xml ${CMAKE_CURRENT_SOURCE_DIR}/models/${MODEL_NAME}.xml)
  file(DOWNLOAD https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/${MODEL_NAME}/FP32/${MODEL_NAME}.bin ${CMAKE_CURRENT_SOURCE_DIR}/models/${MODEL_NAME}.bin)
endif()


# INSTALL ######################################################################
set(CMAKE_INSTALL_PREFIX ${USR_DIR})
install(TARGETS ${TARGET_LIST}
  BUNDLE DESTINATION bin
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
)


