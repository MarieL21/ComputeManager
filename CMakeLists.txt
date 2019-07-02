cmake_minimum_required (VERSION 3.0.2)

set (PROJECT ComputeManagerTest)
project (${PROJECT})

set (${PROJECT}_CUDA_SOURCE
  Cuda/ComputeManager/ComputeManager.cu
  test_cuda.cpp
  )

set (${PROJECT}_CL_SOURCE
  OpenCL/ComputeManager/ComputeManager.cpp
  test_opencl.cpp
  )

# you probably do not need ComputeManager test build
# in your application
option(BUILD_TEST "Build ComputeManager test." OFF)

if (BUILD_TEST)
  find_package(CUDA)
  if (NOT CUDA_FOUND)
	message(WARNING "Could not find NVIDIA CUDA C tools.")
  else()
	CUDA_ADD_EXECUTABLE(test_cuda ${${PROJECT}_CUDA_SOURCE})
  endif()

  find_package(OpenCL)
  if (NOT OpenCL_FOUND)
	message(WARNING "Could not find OpenCL installation.")
  else()
	add_executable(test_opencl ${${PROJECT}_CL_SOURCE})
	target_link_libraries(test_opencl ${OpenCL_LIBRARIES})
  endif()
endif()
  
