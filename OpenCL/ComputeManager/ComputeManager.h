#ifndef _COMPUTEMANAGER_H_
#define _COMPUTEMANAGER_H_

#include "ComputeTypes.h"
#include "kernel_type.h"

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <cassert>
#include <vector>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // turns off gcc warnings on
										  // deprecated api

#ifdef PROJ_CUSTOM_LOGGER
#include "logger.h"
#elif !defined(log_err)
#include <iostream>
#define log_err std::cerr
#define log_endl std::endl
#define log_inf std::cout
#define log_wrn std::cout
#endif

#ifdef WIN32
#	define WHILE_0\
		__pragma( warning(push) )\
		__pragma( warning(disable:4127) )\
		while(0)\
		__pragma( warning(pop) )
#else
#	define WHILE_0 while(0)
#endif

/// Check OpenCL call status and return it if call failed
#define CHECK_OPENCL(ret) \
	do {\
		cl_int clRet = ret; \
		if (clRet != CL_SUCCESS) \
		{\
			log_err << "OpenCL call failed: " << #ret << " returned " << clRet << " on line " << __LINE__ << log_endl; \
			return clRet; \
		}\
	} WHILE_0

#define CHECK_OPENCL_NO_RET(ret) \
	do {\
		cl_int clRet = ret; \
		if (clRet != CL_SUCCESS) \
		{\
			log_err << "OpenCL call failed: " << #ret << " returned " << clRet << " on line " << __LINE__ << log_endl; \
		}\
	} WHILE_0

#define MY_STRINGIFY(A) #A
#define CM_COMPILE_KERNEL(name, build_opts)								  \
    if (ComputeManager::CompileKernelFromStr(_cm->GetContext(), _cm->GetDevice(), \
									  &name,                                      \
									  "kernel_" MY_STRINGIFY(name),		          \
									  name ## _program_source,	                  \
									  build_opts) != CL_SUCCESS){                 \
        throw std::runtime_error("Error in compile_kernel" MY_STRINGIFY(name));   \
    }

/// Cast to cl_mem useful macro
#define ToGpuMem(x) static_cast<cl_mem>(x)

/// @brief Manages OpenCL configuration
///
/// Common way to handle OpenCL initialization and release. Context and device queue are created and stored inside class.
class ComputeManager
{
public:
	ComputeManager()
		: m_device(NULL)
		, m_context(NULL)
		, m_queue(NULL)
		, m_configuredOpenCL(false)
		, m_versionSupported(0)
	{
	}

	~ComputeManager()
	{
		Release();
	}

	static cl_int GetComputeDevices(std::vector<ComputeDeviceLib> &devices);
	cl_int GetCurrentComputeDevice(ComputeDeviceLib &currentDevice);
	cl_int SetComputeDevice(ComputeDeviceLib deviceToUse);

	cl_int Initialize() const;
	cl_int Release();

	cl_device_id     GetDevice()          const {return m_device;}
	cl_context       GetContext()         const {return m_context;}
	cl_command_queue GetQueue()           const {return m_queue;}
	cl_command_queue* GetQueuePtr()       const {return &m_queue;}		
	
	cl_uint          GetStandardVersion() const {return m_versionSupported;}

	static cl_int GetPlatformOpenCLVersion(cl_device_id device, int &major, int &minor);
	static cl_int ChooseBestDeviceAvailable(cl_device_id &device);
	static double GetEventTime(cl_event event);

	static cl_int CompileKernelFromStr(cl_context context, cl_device_id device,
									   cl_kernel_t *kernel, const char* source_str_ptr,
									   const char* filename, const char* build_options = NULL);	

	static cl_int ReleaseKernelAndProgram(cl_kernel_t& a_kernel);
	static bool CheckNoErrIsSameAsApi(int app_success);

protected:
	mutable cl_device_id     m_device;            ///< OpenCL device on which queue is executed
	mutable cl_context       m_context;           ///< OpenCL context
	mutable cl_command_queue m_queue;             ///< OpenCL command queue
	mutable bool             m_configuredOpenCL;  ///< If OpenCL context, queue, devices were succesfully set up
	mutable cl_uint          m_versionSupported;  ///< OpenCL context supported version, encoded in decimal number - 10 for 1.0, 11 for 1.1, 12 for 1.2
};

#endif // _COMPUTEMANAGER_H_
