#   ____  _             _       
#  |  _ \| |_   _  __ _(_)_ __  
#  | |_) | | | | |/ _` | | '_ \ 
#  |  __/| | |_| | (_| | | | | |
#  |_|   |_|\__,_|\__, |_|_| |_|
#                 |___/         
cmake_minimum_required(VERSION 3.20)
project(plugin VERSION 0.1.0 LANGUAGES CXX)


# PROJECT SETTINGS #############################################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(USR_DIR ${CMAKE_CURRENT_LIST_DIR}/usr)
set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src)
set(CMAKE_INSTALL_PREFIX ${USR_DIR})
set(FETCHCONTENT_QUIET OFF)
option(SKIP_EXTERNAL "Skip external dependencies" OFF)


# DEPENDENCIES #################################################################
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
set(BUILD_TESTING OFF CACHE INTERNAL "")
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pugg json)


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

include_directories(
  ${json_SOURCE_DIR}/include
  ${USR_DIR}/include
)
link_directories(
  ${USR_DIR}/lib
)
if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_compile_options(/wd4068 /wd4005 /wd4267 /wd4244 /wd4804 /EHsc)
else()
  add_compile_options(-Wno-deprecated-declarations)
endif()

if(WIN32)
  set(ARMA_LIBS 
    armadillo 
    ${EXTERNAL_DIR}/armadillo/examples/lib_win64/libopenblas.lib
  )
else()
  set(ARMA_LIBS armadillo)
endif()


# TARGETS ######################################################################
add_plugin(echo)
add_plugin(echoj)
add_plugin(twice)
add_plugin(clock)
add_plugin(force_platform LIBS ${ARMA_LIBS})


add_loader(loader)
add_loader(load_filter)
add_loader(load_source)

find_package(OpenCV)
if(OpenCV_FOUND)
  message(STATUS "OpenCV found, webcam.plugin will be built")
  add_plugin(webcam LIBS ${OpenCV_LIBS})
endif()

if (NOT SKIP_EXTERNAL)
  include(external.cmake)
  add_dependencies(force_platform mlpack)
endif()



# INSTALL ######################################################################
install(TARGETS ${TARGET_LIST}
  BUNDLE DESTINATION bin
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
)


# TESTING ######################################################################
include(CTest)
add_test(NAME "echoj plugin exec" COMMAND ${echoj_EXEC})
add_test(NAME "clock plugin exec" COMMAND ${clock_EXEC})
if (APPLE)
  add_test(NAME "load_filter echoj.plugin" COMMAND load_filter echoj.plugin)
  add_test(NAME "load_source clock.plugin" COMMAND load_source clock.plugin)
elseif(WIN32)
  add_test(NAME "load_filter echoj.plugin" COMMAND load_filter.exe echoj.plugin)
  add_test(NAME "load_source echoj.plugin" COMMAND load_source.exe clock.plugin)
else()
  add_test(NAME "load_filter echoj.plugin" COMMAND build/load_filter build/echoj.plugin WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
  add_test(NAME "load_source echoj.plugin" COMMAND build/load_source build/clock.plugin WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
endif()