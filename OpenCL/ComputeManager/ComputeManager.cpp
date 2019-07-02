#include "ComputeManager.h"

#include <cstdio>
#include <cstring>
#include <fstream>


/// Get OpenCL version supported by platform. For platform or context related version-specific functions.
/// @param[in]  device  device of tested platform
/// @param[out] major   major OpenCL version number - 1 for 1.2
/// @param[out] minor   minor OpenCL version number - 2 for 1.2
cl_int ComputeManager::GetPlatformOpenCLVersion(cl_device_id device, int &major, int &minor)
{
	cl_int clStatus = CL_SUCCESS;
#ifndef BOX_SOURCE_VERSION
	cl_platform_id platform;
	clStatus = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, NULL);
	if (CL_SUCCESS != clStatus)
	{
		return clStatus;
	}
	char platformVersion[256];
	clStatus = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(platformVersion), platformVersion, NULL);
	if (CL_SUCCESS != clStatus)
	{
		return clStatus;
	}
	// Anyway, we have OpenCL 1.0 at least
	major = 1;
	minor = 0;
	const char *format = "OpenCL %d.%d ";
#ifdef _MSC_VER
	sscanf_s(platformVersion, format, &major, &minor);
#else
	sscanf(platformVersion, format, &major, &minor);
#endif
#endif
	return clStatus;
}


cl_int ComputeManager::CompileKernelFromStr(cl_context context, cl_device_id device,
											cl_kernel_t *kernel,
											const char* function_name,
											const char* source_str_ptr,
											const char* build_options)
{
	int ret;	
    kernel->program = clCreateProgramWithSource(context, 1, (const char**)&source_str_ptr, NULL, &ret);
	CHECK_OPENCL(ret);	
    ret = clBuildProgram(kernel->program, 1, &device, build_options, NULL, NULL);
	if (ret != CL_SUCCESS)
	{
		// Determine the size of the log
		size_t build_log_size;
		clGetProgramBuildInfo(kernel->program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);

		// Allocate memory for the log
		std::string build_log;
		build_log.resize(build_log_size);		
		if (!build_log.empty())
		{			
			// Get the log
			clGetProgramBuildInfo(kernel->program, device, CL_PROGRAM_BUILD_LOG, build_log_size, &build_log[0], NULL);

			// Print the log
			log_inf << build_log << log_endl;
		}	
	}
	CHECK_OPENCL(ret);	
    kernel->kernel = clCreateKernel(kernel->program, function_name, &ret);	
    return ret;	
}


/// Get list of available OpenCL devices
/// @return @c CL_SUCCESS if everything is fine, OpenCL error code otherwise
cl_int ComputeManager::GetComputeDevices(std::vector<ComputeDeviceLib> &devices)
{
	cl_int clStatus = CL_SUCCESS;
#ifndef BOX_SOURCE_VERSION
	cl_uint platformsNumber = 0;
	clStatus = clGetPlatformIDs(0, NULL, &platformsNumber);
	if (CL_PLATFORM_NOT_FOUND_KHR == clStatus)
	{
		log_err << "No icd-compatible OpenCL platforms found" << log_endl;
		return clStatus;
	}
	CHECK_OPENCL(clStatus);
	if (platformsNumber == 0)
	{
		// this should not happen by standard, but check it anyway
		return CL_PLATFORM_NOT_FOUND_KHR;
	}

	cl_platform_id *platforms = new cl_platform_id[platformsNumber];
	if (!platforms)
	{
		// no memory
		return CL_OUT_OF_HOST_MEMORY;
	}

	CHECK_OPENCL(clGetPlatformIDs(platformsNumber, platforms, NULL));

	for (cl_uint platformIndex = 0; platformIndex < platformsNumber; platformIndex++)
	{
		char platformName[256]={0};
		cl_platform_id platform = platforms[platformIndex];
		CHECK_OPENCL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL));
		log_inf << "Found OpenCL platform: " << platformName << log_endl;

		cl_uint devicesNumber = (cl_uint)(-1);
		clStatus = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &devicesNumber);
		if ((CL_DEVICE_NOT_FOUND == clStatus) || (0 == devicesNumber))
		{
			clStatus = CL_SUCCESS;
			continue;
		}
		cl_device_id *platformDevices = new cl_device_id[devicesNumber];
		if (!platformDevices)
		{
			// no memory
			delete[] platforms;
			return CL_OUT_OF_HOST_MEMORY;
		}
		CHECK_OPENCL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesNumber, platformDevices, NULL));
		for (cl_uint deviceIndex = 0; deviceIndex < devicesNumber; deviceIndex++)
		{
			char deviceName[256]={0};
			cl_device_id device = platformDevices[deviceIndex];
			CHECK_OPENCL(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL));
			log_inf << "Found OpenCL device: " << deviceName << log_endl;

			ComputeDeviceLib dev(platformName, deviceName, device);
			devices.push_back(dev);
		}
		delete[] platformDevices;
	}
	delete[] platforms;
#endif
	return clStatus;
}

/// Set OpenCL device to use
/// @return @c CL_SUCCESS if everything is fine, OpenCL error code otherwise
cl_int ComputeManager::SetComputeDevice(ComputeDeviceLib deviceToUse)
{
#ifndef BOX_SOURCE_VERSION
	cl_device_id newDevice = reinterpret_cast<cl_device_id>(deviceToUse.deviceId);
	if (m_device != newDevice) // no need to reallocate queue and context if device was not changed
	{
		if (m_configuredOpenCL)
		{
			CHECK_OPENCL(Release());
		}
		m_device = newDevice;
	}
#endif
	return CL_SUCCESS;
}
/// Get current OpenCL device struct - device pointer, device name and platform name
/// @return @c CL_SUCCESS if everything is fine, OpenCL error code otherwise
cl_int ComputeManager::GetCurrentComputeDevice(ComputeDeviceLib &currentDevice)
{
#ifndef BOX_SOURCE_VERSION
	if (NULL == m_device)
	{
		// SetDevice() was not called, select some device
		CHECK_OPENCL(ChooseBestDeviceAvailable(m_device));
	}

	cl_platform_id platform = NULL;
	char platformName[256]={0};
	char deviceName[256]={0};
	CHECK_OPENCL(clGetDeviceInfo(m_device, CL_DEVICE_PLATFORM, sizeof(platform), &platform, NULL));
	CHECK_OPENCL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL));
	CHECK_OPENCL(clGetDeviceInfo(m_device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL));
	ComputeDeviceLib dev(platformName, deviceName, m_device);
	currentDevice = dev;
#endif
	return CL_SUCCESS;
}

enum PlatformType
{
	PLATFORM_UNKNOWN,
	PLATFORM_AMD,
	PLATFORM_INTEL,
	PLATFORM_NVIDIA
};

/// @brief Select best available OpenCL device
///
/// For now, best means GPU device with maximum performance in GFLOPS. Performance calculations for
/// AMD and Intel seems to be accurate, for NVidia they are semi-accurate depending on GPU model.
/// @return @c CL_SUCCESS if everything is fine, OpenCL error code otherwise
cl_int ComputeManager::ChooseBestDeviceAvailable(cl_device_id &device)
{
	cl_int clStatus = CL_SUCCESS;
#ifndef BOX_SOURCE_VERSION
	cl_uint platformsNumber = 0;
	clStatus = clGetPlatformIDs(0, NULL, &platformsNumber);
	if (CL_PLATFORM_NOT_FOUND_KHR == clStatus)
	{
		log_err << "No icd-compatible OpenCL platforms found" << log_endl;
		return clStatus;
	}
	CHECK_OPENCL(clStatus);
	if (platformsNumber == 0)
	{
		// this should not happen by standard, but check it anyway
		return CL_PLATFORM_NOT_FOUND_KHR;
	}

	cl_platform_id *platforms = new cl_platform_id[platformsNumber];
	if (!platforms)
	{
		// no memory
		return CL_OUT_OF_HOST_MEMORY;
	}

	CHECK_OPENCL(clGetPlatformIDs(platformsNumber, platforms, NULL));

	cl_float maxDeviceRating = 0;
	cl_device_id bestDevice = device;
	for (cl_uint platformIndex = 0; platformIndex < platformsNumber; platformIndex++)
	{
		cl_uint devicesNumber = (cl_uint)(-1);
		clStatus = clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_ALL, 0, NULL, &devicesNumber);
		if ((CL_DEVICE_NOT_FOUND == clStatus) || (0 == devicesNumber))
		{
			clStatus = CL_SUCCESS;
			continue;
		}

		char platformName[512] = {0};
		CHECK_OPENCL(clGetPlatformInfo(platforms[platformIndex], CL_PLATFORM_NAME, 512, platformName, NULL));
		bool isOcland = strstr(platformName, "ocland");

		PlatformType ptype = PLATFORM_UNKNOWN;
		if (strstr(platformName, "AMD "))
		{
			ptype = PLATFORM_AMD;
		}
		else if (strstr(platformName, "Intel(R) "))
		{
			ptype = PLATFORM_INTEL;
		}
		else if (strstr(platformName, "NVIDIA "))
		{
			ptype = PLATFORM_NVIDIA;
		}
		cl_device_id *platformDevices = new cl_device_id[devicesNumber];
		if (!platformDevices)
		{
			// no memory
			delete[] platforms;
			return CL_OUT_OF_HOST_MEMORY;
		}
		CHECK_OPENCL(clGetDeviceIDs(platforms[platformIndex], CL_DEVICE_TYPE_ALL, devicesNumber, platformDevices, NULL));
		for (cl_uint deviceIndex = 0; deviceIndex < devicesNumber; deviceIndex++)
		{
			cl_float deviceRating = 0;
			cl_uint computeUnitsNumber = 0;
			cl_uint clockFreq = 0;
			cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
			CHECK_OPENCL(clGetDeviceInfo(platformDevices[deviceIndex], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnitsNumber), &computeUnitsNumber, NULL));
			deviceRating = computeUnitsNumber;
			cl_int ret = clGetDeviceInfo(platformDevices[deviceIndex], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFreq), &clockFreq, NULL);
			// workaround for wrong frequency on some platforms:
			if ((CL_SUCCESS != ret) || (0 == clockFreq))
			{
				if (CL_SUCCESS != ret)
				{
					// returns -5 (CL_OUT_OF_RESOURCES) sometimes on NVidia 358.87 driver
					log_wrn << "OpenCL call failed: clGetDeviceInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY) returned " << ret
						<< ", assuming 1000 MHz."<< log_endl;
				}
				else if (0 == clockFreq)
				{
					// Happens on Intel CPU OpenCL working on AMD processor
					log_wrn << "Zero compute device frequency returned, assuming 1000 MHz." << log_endl;
				}
				clockFreq = 1000;
			}
			deviceRating *= clockFreq; // clock iz in MHz

			CHECK_OPENCL(clGetDeviceInfo(platformDevices[deviceIndex], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL));
			if (deviceType & CL_DEVICE_TYPE_CPU)
			{
				// CPU cores are less powerful than GPU compute units
				if (PLATFORM_INTEL == ptype)
				{
					deviceRating *= (2 * 4); // fma on 128-bit AVX
				}
				else
				{
					deviceRating *= 2; // assume simple fma is supported
				}
			}
			else if (deviceType & CL_DEVICE_TYPE_GPU)
			{
				switch (ptype) {
					case PLATFORM_AMD:
						// This one is precise peak performance of AMD device CU per clock really.
						deviceRating *= 2 * 64; // 64 processing elements per CU, each does fma
						break;
					case PLATFORM_INTEL:
						// Should be true on all Intel GPU devices, but maybe there are some exceptions.
						deviceRating *= 16;
						break;
					case PLATFORM_NVIDIA:
						// This could be false depending on exact NVidia device model
						// but good enouth for a guess.
						deviceRating *= 256;
						break;
					default:
						deviceRating *= 2; // assume each CU does single fma per clock
						break;
				}
			}
			deviceRating /= 1000.0f;
			// deviceRating contains peak device performance in GFLOPS now

			if (isOcland)
			{
				// Use ocland if we configured it for the application,
				// even if have device with same rating on host.
				// If you do not want ocland to be used in such case, just do not enable ocland.
				deviceRating++;
			}

			char deviceName[256]={0};
			CHECK_OPENCL(clGetDeviceInfo(platformDevices[deviceIndex], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL));
			log_inf << "device: " << deviceName << " " << deviceRating << " GFLOPS" << log_endl;

			if (deviceRating > maxDeviceRating)
			{
				maxDeviceRating = deviceRating;
				bestDevice = platformDevices[deviceIndex];
			}
		}
		delete[] platformDevices;
	}
	delete[] platforms;

	device = bestDevice;
#endif
	return clStatus;
}

/// Get OpenCL event start/end time in milliseconds
///
/// @return time in milliseconds or negative error code in case of failure
double ComputeManager::GetEventTime(cl_event event)
{
#ifdef BOX_SOURCE_VERSION
	return 0.0;
#else
	cl_ulong eventStartTime = 0;
	cl_ulong eventEndTime = 0;

	CHECK_OPENCL(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(eventStartTime),
			&eventStartTime, NULL));
	CHECK_OPENCL(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(eventEndTime),
			&eventEndTime, NULL));
	return 1e-6 * (eventEndTime - eventStartTime);
#endif
}


/// Select device, initialize OpenCL context, create command queue
/// @return @c CL_SUCCESS if everything is fine, OpenCL error code otherwise
cl_int ComputeManager::Initialize() const
{
#ifndef BOX_SOURCE_VERSION
	cl_int clStatus = CL_SUCCESS;

	if (m_configuredOpenCL)
	{
		log_inf << "OpenCL has already been configured" << log_endl;
		return clStatus;
	}

	if (NULL == m_device)
	{
		// SetDevice() was not called, select some device
		CHECK_OPENCL(ChooseBestDeviceAvailable(m_device));
		if(NULL == m_device)
		{
			// we have some OpenCL platforms but no apropriate devices
			log_err << "No OpenCL devices found" << log_endl;
			return CL_DEVICE_NOT_FOUND;
		}
	}
	char deviceName[256]={0};
	CHECK_OPENCL(clGetDeviceInfo(m_device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL));
	log_inf << "Using OpenCL device: " << deviceName << " [" << m_device << "]" << log_endl;

	m_context = clCreateContext(NULL, 1, &m_device, NULL, NULL, &clStatus);
	CHECK_OPENCL(clStatus);

	#ifdef NDEBUG
	m_queue = clCreateCommandQueue(m_context, m_device, 0, &clStatus);
	#else
	m_queue = clCreateCommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &clStatus);
	#endif
	
	CHECK_OPENCL(clStatus);

	int versionMajor = 0;
	int versionMinor = 0;
	CHECK_OPENCL(GetPlatformOpenCLVersion(m_device, versionMajor, versionMinor));
	// We suppose minor version lower than 10 always. It is so for all current (1.0, 1.1, 1.2)
	// and known future (2.0) OpenCL standard versions.
	m_versionSupported = versionMajor * 10 + versionMinor;

	if (m_versionSupported >= 12)
	{
		log_inf << "OpenCL 1.2 is supported (current version is " << versionMajor << "." << versionMinor << ")" << log_endl;
	}
	else
	{
		log_inf << "OpenCL 1.2 is NOT supported (current version is " << versionMajor << "." << versionMinor << ")" << log_endl;
	}

	if (clStatus == CL_SUCCESS)
	{
		m_configuredOpenCL = true;
	}
	return clStatus;
#endif
}

cl_int ComputeManager::ReleaseKernelAndProgram(cl_kernel_t& a_kernel)
{
	cl_int clStatus;	
	clStatus = clReleaseKernel(a_kernel.kernel);
	CHECK_OPENCL_NO_RET(clStatus);	

    clStatus = clReleaseProgram(a_kernel.program);
	CHECK_OPENCL_NO_RET(clStatus);

	return clStatus;	
}

/// Release OpenCL objects acquired earlier - command queue and context
/// @return @c CL_SUCCESS if everything is fine, OpenCL error code otherwise
cl_int ComputeManager::Release()
{
#ifndef BOX_SOURCE_VERSION
	if (m_queue)
	{		
		CHECK_OPENCL(clReleaseCommandQueue(m_queue));
		m_queue = NULL;
	}
	if (m_context)
	{
		CHECK_OPENCL(clReleaseContext(m_context));
		m_context = NULL;
	}
	m_versionSupported = 0;
	m_device = NULL;
	m_configuredOpenCL = false;
#endif
	return CL_SUCCESS;
}

bool ComputeManager::CheckNoErrIsSameAsApi(int app_success)
{
	return app_success == CL_SUCCESS;	
}
