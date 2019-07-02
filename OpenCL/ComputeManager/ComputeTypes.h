#ifndef COMPUTETYPES_H_
#define COMPUTETYPES_H_

#include <string>

/// Identifier of OpenCL compute device - platfofm name and device name
class ComputeDeviceLib
{
public:
	typedef void* DeviceIdType;	
	
	ComputeDeviceLib(): platformName(), deviceName(), deviceId(NULL){}

	ComputeDeviceLib(std::string platformNameString, std::string deviceNameString, void* deviceIdentifier)
		: platformName(platformNameString)
		, deviceName(deviceNameString)
		, deviceId(deviceIdentifier)
	{}

	std::string platformName; ///< full name of OpenCL platform
	std::string deviceName;   ///< full name of OpenCL device
	DeviceIdType deviceId;     ///< OpenCL device id
};

#endif /* COMPUTETYPES_H_ */
