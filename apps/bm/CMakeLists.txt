cmake_minimum_required(VERSION 3.30)
project(bm)

######################################################################
## Required C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bm src/bm.cpp)
target_link_libraries(bm diskpp::diskpp)
install(TARGETS bm RUNTIME DESTINATION bin)
