project( LocalSearchExample )
cmake_minimum_required( VERSION 2.8 )

find_package( CUDA REQUIRED )

SET( CMAKE_BUILD_TYPE "Release")
SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -fopenmp" )

FILE(MAKE_DIRECTORY "bin")


FOREACH(directory IN ITEMS ls_on_cpu_sequential ls_on_cpu_openmp_1 ls_on_cpu_openmp_2)
   FILE( GLOB src "${directory}/main.cpp" "${directory}/ls_example_cpu*.cpp" )
   CUDA_ADD_EXECUTABLE( "bin/${directory}" ${src} )
ENDFOREACH(directory)


INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS} ./include)

SET(CUDA_NVCC_FLAGS "-arch=compute_10;-code=sm_10;")
SET(CUDA_ARCH_BIN "10")
FOREACH(directory IN ITEMS ls_on_gpu_1 ls_on_gpu_2 ls_on_gpu_3 ls_on_gpu_4 ls_on_gpu_5 ls_on_gpu_6 ls_on_gpu_8 ls_on_gpu_9 ls_on_gpu_filter_1 ls_on_gpu_filter_2 ls_on_gpu_filter_3 ls_on_gpu_filter_4)
   FILE( GLOB src "${directory}/main.cpp" "${directory}/ls_example_gpu*.cu" )
   CUDA_ADD_EXECUTABLE( "bin/${directory}__cc10" ${src} )
ENDFOREACH(directory)

SET(CUDA_NVCC_FLAGS "-arch=compute_20;-code=sm_20;")
SET(CUDA_ARCH_BIN "20")
FOREACH(directory IN ITEMS ls_on_gpu_1 ls_on_gpu_2 ls_on_gpu_3 ls_on_gpu_4 ls_on_gpu_5 ls_on_gpu_6 ls_on_gpu_8 ls_on_gpu_9 ls_on_gpu_filter_1 ls_on_gpu_filter_2 ls_on_gpu_filter_3 ls_on_gpu_filter_4)
   FILE( GLOB src "${directory}/main.cpp" "${directory}/ls_example_gpu*.cu" )
   CUDA_ADD_EXECUTABLE( "bin/${directory}__cc20" ${src} )
ENDFOREACH(directory)
