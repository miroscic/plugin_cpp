cmake_minimum_required(VERSION 3.20)
project(plugin VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(USR_DIR ${CMAKE_CURRENT_LIST_DIR}/usr)
set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src)

message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_LIST_DIR: ${CMAKE_LIST_DIR}")
message(STATUS "USR_DIR: ${USR_DIR}")
message(STATUS "SRC_DIR: ${SRC_DIR}")

include(FetchContent)
FetchContent_Declare(pugg 
  GIT_REPOSITORY https://github.com/pbosetti/pugg.git
  GIT_TAG        1.0.2
  GIT_SHALLOW    TRUE
)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pugg json)

include_directories(${json_SOURCE_DIR}/include)

macro(add_plugin name)
  # on MacOS only, plugins can be compiled as executables
  if (APPLE)
    add_executable(${name} ${SRC_DIR}/plugin/${name}.cpp)
    set_target_properties(${name} PROPERTIES ENABLE_EXPORTS TRUE)
    set(${name}_EXEC ${name}.plugin)
  else()
    add_library(${name} SHARED ${SRC_DIR}/plugin/${name}.cpp)
    add_executable(${name}_main ${SRC_DIR}/plugin/${name}.cpp)
    target_link_libraries(${name}_main PRIVATE pugg)
    set(${name}_EXEC ${name})
    set_target_properties(${name}_main PROPERTIES OUTPUT_NAME ${name})
    list(APPEND TARGET_LIST ${name}_main)
  endif()
  target_link_libraries(${name} PRIVATE pugg)
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

add_plugin(echo)
add_plugin(echoj)
add_plugin(twice)
add_plugin(clock)

add_loader(loader)
add_loader(load_filter)
add_loader(load_source)


set(CMAKE_INSTALL_PREFIX ${USR_DIR})
install(TARGETS ${TARGET_LIST}
  BUNDLE DESTINATION bin
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
)

include(CTest)
add_test(NAME "echoj plugin exec" COMMAND ${echoj_EXEC})
add_test(NAME "clock plugin exec" COMMAND ${clock_EXEC})
if (APPLE)
  add_test(NAME "load_filter echoj.plugin" COMMAND load_filter echoj.plugin)
  add_test(NAME "load_filter clock.plugin" COMMAND load_source clock.plugin)
elseif(WIN32)
  add_test(NAME "load_filter echoj.plugin" COMMAND load_filter.exe echoj.plugin)
  add_test(NAME "load_filter echoj.plugin" COMMAND load_source.exe clock.plugin)
else()
  add_test(NAME "load_filter echoj.plugin" COMMAND build/load_filter build/echoj.plugin WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
  add_test(NAME "load_filter echoj.plugin" COMMAND build/load_source build/clock.plugin WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
endif()