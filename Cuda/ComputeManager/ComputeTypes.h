#ifndef COMPUTE_TYPES_CUDA_H_
#define COMPUTE_TYPES_CUDA_H_

#include <string>
#include <stdint.h>

class ComputeDeviceLib
{
  public:
	typedef int64_t DeviceIdType;	

  ComputeDeviceLib()
	  :platformName("NVIDIA CUDA")
		,deviceName()
		,deviceId(0)
	{
	}

  ComputeDeviceLib(std::string deviceNameString, int deviceIdentifier)
	  :platformName("NVIDIA CUDA")
		,deviceName(deviceNameString)
		,deviceId(deviceIdentifier)
	{
	}
	std::string platformName; ///< full name of CUDA platform
	std::string deviceName;   ///< full name of CUDA device
	DeviceIdType deviceId;     ///< CUDA device id
};

#endif // COMPUTE_TYPES_CUDA_H_
