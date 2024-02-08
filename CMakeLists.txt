cmake_minimum_required(VERSION 3.20)
project(plugin VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
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
  add_library(${name} SHARED 
    ${SRC_DIR}/plugin/${name}.cpp
  )
  target_link_libraries(${name} PRIVATE pugg)
  set_target_properties(${name} PROPERTIES PREFIX "")
  set_target_properties(${name} PROPERTIES SUFFIX ".plugin")
  list(APPEND TARGET_LIST ${name})
endmacro()

macro (add_plugin_and_exe name)
  add_plugin(${name})
  add_executable(${name}_test ${SRC_DIR}/plugin/${name}.cpp)
  target_compile_definitions(${name}_test PRIVATE HAVE_MAIN)
  list(APPEND TARGET_LIST ${name}_test)
endmacro()

add_plugin(echo)
add_plugin_and_exe(echoj)
add_plugin_and_exe(twice)

add_executable(loader ${SRC_DIR}/main/loader.cpp)
target_link_libraries(loader PRIVATE pugg)
list(APPEND TARGET_LIST loader)

add_executable(loaderj ${SRC_DIR}/main/loaderj.cpp)
target_link_libraries(loaderj PRIVATE pugg)
list(APPEND TARGET_LIST loaderj)



set(CMAKE_INSTALL_PREFIX ${USR_DIR})
install(TARGETS ${TARGET_LIST}
  BUNDLE DESTINATION bin
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
)