#ifndef COMPUTE_MANAGER_CUDA_H_
#define COMPUTE_MANAGER_CUDA_H_

#include <vector>
#include "ComputeTypes.h"

// includes CUDA Runtime
#include <cuda_runtime.h>

#ifdef PROJ_CUSTOM_LOGGER
#include "logger.h"
#elif !defined(log_err)
#include <iostream>
#define log_err std::cerr
#define log_endl std::endl
#define log_inf std::cout
#define log_wrn std::cout
#endif

/// Check Cuda call status and return it if call failed
#define CHECK_CUDA(ret) \
	do {\
		cudaError_t cuRet = ret; \
		if (cuRet != cudaSuccess) \
		{\
			log_err << "Cuda call failed: " << #ret << " returned " << cudaGetErrorName(cuRet) << " on line " << __LINE__ \
					<< ", which means: " << cudaGetErrorString(cuRet) << log_endl; \
			return cuRet; \
		}\
	} while(0)

#define CHECK_CUFFT(ret) \
	do {\
		cufftResult_t cuRet = ret; \
		if (cuRet != CUFFT_SUCCESS) \
		{\
			log_err << "CUFFT call failed: " << #ret << " returned " << cuRet << " on line " << __LINE__ \
					<< log_endl; \
			return cuRet; \
		}\
	} while(0)

#define CHECK_CUFFT_NO_RET(ret) \
	do {\
		cufftResult_t cuRet = ret; \
		if (cuRet != CUFFT_SUCCESS) \
		{\
			log_err << "CUFFT call failed: " << #ret << " returned " << cuRet << " on line " << __LINE__ \
					<< log_endl; \
		}\
	} while(0)

#define CHECK_CUDA_NO_RET(ret) \
	do {\
		cudaError_t cuRet = ret; \
		if (cuRet != cudaSuccess) \
		{\
			log_err << "Cuda call failed: " << #ret << " returned " << cudaGetErrorName(cuRet) << " on line " << __LINE__ \
					<< ", which means: " << cudaGetErrorString(cuRet) << log_endl; \
		}\
	} while(0)

#define ToGpuMem(x) x

/// @brief Manages Cuda configuration
class ComputeManager
{
public:
	ComputeManager();	
	
	~ComputeManager()
	{
		Release();
	}
	cudaError_t Release();		
	static cudaError_t GetComputeDevices(std::vector<ComputeDeviceLib> &devices);
	cudaError_t SetComputeDevice(ComputeDeviceLib deviceToUse);
	cudaError_t Initialize() const;
	int GetSmCount() const;
	static bool CheckNoErrIsSameAsApi(int app_success);	
  private:
	cudaError_t ChooseBestDeviceAvailable(int& device_id) const;	
	
	mutable int device_id;
	mutable int sm_count;	
};

#endif // COMPUTE_MANAGER_CUDA_H_
