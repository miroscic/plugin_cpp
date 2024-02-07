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
FetchContent_MakeAvailable(pugg)



add_library(echo SHARED 
  ${SRC_DIR}/plugin/echo.cpp
)
target_link_libraries(echo PRIVATE pugg)
set_target_properties(echo PROPERTIES PREFIX "")
set_target_properties(echo PROPERTIES SUFFIX ".plugin")
list(APPEND TARGET_LIST echo)

add_library(twice SHARED 
  ${SRC_DIR}/plugin/twice.cpp
)
target_link_libraries(twice PRIVATE pugg)
set_target_properties(twice PROPERTIES PREFIX "")
set_target_properties(twice PROPERTIES SUFFIX ".plugin")
list(APPEND TARGET_LIST twice)

add_executable(loader ${SRC_DIR}/main/loader.cpp)
target_link_libraries(loader PRIVATE pugg)
list(APPEND TARGET_LIST loader)

add_executable(twice_test ${SRC_DIR}/plugin/twice.cpp)
target_compile_definitions(twice_test PRIVATE HAVE_MAIN)
list(APPEND TARGET_LIST twice_test)



set(CMAKE_INSTALL_PREFIX ${USR_DIR})
install(TARGETS ${TARGET_LIST}
  BUNDLE DESTINATION bin
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  ARCHIVE DESTINATION lib
)