cmake_minimum_required(VERSION 3.15)
project(VulkanProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Look for Vulkan using native system paths
find_package(Vulkan REQUIRED)

add_executable(VulkanProject main.cpp)

# Attach includes and libraries seamlessly 
target_link_libraries(VulkanProject PRIVATE Vulkan::Vulkan)
