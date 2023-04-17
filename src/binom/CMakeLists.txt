
cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost COMPONENTS math) 
include_directories( ${Boost_INCLUDE_DIR} )

add_library(binom SHARED ${CMAKE_CURRENT_SOURCE_DIR}/binom.cpp)
set_target_properties(binom PROPERTIES LINKER_LANGUAGE C)
