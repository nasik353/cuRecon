cmake_minimum_required(VERSION 3.10)

project(cuRecon CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DNOMINMAX) 

find_package(fmt CONFIG REQUIRED)
find_package(HDF5 CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(CUDA REQUIRED)
find_package(Thrust REQUIRED)
find_package(OpenGL REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)

include_directories(include)

file(GLOB_RECURSE SOURCES 
    "${CMAKE_SOURCE_DIR}/src/**/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/**/*.cu"
)

# add_executable(cuRecon_demo ${SOURCES_DEMO})
add_executable(cuRecon ${SOURCES})

# target_compile_definitions(cuRecon_demo PRIVATE GLEW_STATIC)

# Include directories
target_include_directories(cuRecon PRIVATE
    ${HDF5_INCLUDE_DIR}
)


# Link libraries
target_link_libraries(cuRecon PRIVATE
    fmt::fmt
    ${CUDA_LIBRARIES}
    hdf5::hdf5-static hdf5::hdf5_hl-static
    cxxopts::cxxopts
    Thrust::Thrust
    OpenGL::GL
    imgui::imgui
    glfw
    GLEW::GLEW
)

# Enable separate compilation for CUDA
set_property(TARGET cuRecon PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Set C++ compiler flags for CUDA
set_target_properties(cuRecon PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuRecon PROPERTIES CUDA_ARCHITECTURES "75")
