#include "ComputeManager.h"

#if (CUDART_VERSION <= 6050)
int cudaInvalidDeviceId = -1;
#endif

int GetSPcores(const cudaDeviceProp& devProp);

int IsValidDevice(int device_id)
{
	return device_id != cudaInvalidDeviceId;	
}

ComputeManager::ComputeManager()
:device_id(cudaInvalidDeviceId),
 sm_count(2),
 is_initialized(true)	
{	
}

cudaError_t ComputeManager::GetComputeDevices(std::vector<ComputeDeviceLib> &devices)
{
	int device_count = 0;
	CHECK_CUDA(cudaGetDeviceCount(&device_count));
	if (!device_count)
	{
		log_err << "No CUDA enabled devices found." << log_endl;
		return cudaSuccess;		
	}
	devices.resize(device_count);
	cudaDeviceProp prop;	
	for (int i = 0; i < device_count; ++i)
	{
		devices[i].deviceId = i;
		CHECK_CUDA(cudaGetDeviceProperties(&prop, i));		
		devices[i].deviceName = std::string(prop.name);
	}	
	return cudaSuccess;	
}

bool ComputeManager::CheckNoErrIsSameAsApi(int app_success)
{
	return app_success == cudaSuccess;	
}

cudaError_t ComputeManager::Release()
{
	device_id = cudaInvalidDeviceId;	
	return cudaSuccess;	
}

cudaError_t ComputeManager::SetComputeDevice(ComputeDeviceLib deviceToUse)
{
	if (IsValidDevice(deviceToUse.deviceId))
	{
		CHECK_CUDA(cudaSetDevice(deviceToUse.deviceId));
		device_id = deviceToUse.deviceId;		
		return cudaSuccess;
	}
	else
	{
		log_err << "Invalid deviceId." << log_endl;		
	}
	return cudaErrorInvalidValue;	
}

cudaError_t ComputeManager::Initialize() const
{
	if (IsValidDevice(device_id))
	{
		log_inf << "CUDA device has already been configured!" << log_endl;
		return cudaSuccess;		
	}
	CHECK_CUDA(ChooseBestDeviceAvailable(device_id));
	if (IsValidDevice(device_id))
	{
		cudaDeviceProp prop;
		CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
		log_inf << "Using CUDA device: " << prop.name << " [" << device_id <<  "]"
			   ", compute capability: " << prop.major << "." << prop.minor
				<< log_endl;
		sm_count = prop.multiProcessorCount;
		CHECK_CUDA(cudaSetDevice(device_id));		
	}
	is_initialized = true;
	return cudaSuccess;
}

bool ComputeManager::IsInitialized() const
{
	return is_initialized;	
}

int ComputeManager::GetSmCount() const
{
	return sm_count;	
}

cudaError_t ComputeManager::ChooseBestDeviceAvailable(int& device_id) const
{
	std::vector<ComputeDeviceLib> devices;
	CHECK_CUDA(GetComputeDevices(devices));

	cudaDeviceProp prop;
	float max_device_rating = 0;
	int best_device = cudaInvalidDeviceId;	
	for (int i = 0; i < (int)devices.size(); ++i)
	{		
		CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
		float curr_device_rating = 2.0 * prop.clockRate * GetSPcores(prop)/1e6;
		if (curr_device_rating > max_device_rating)
		{
			max_device_rating = curr_device_rating;
			best_device = i;			
		}
		log_inf << "device: " << devices[i].deviceName << " "
				<< curr_device_rating << " GFLOPS" << log_endl;
	}
	device_id = best_device;
	return cudaSuccess;	
}

int GetSPcores(const cudaDeviceProp& devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major)
	{
		case 2: // Fermi
			if (devProp.minor == 1) cores = mp * 48;
			else cores = mp * 32;
			break;
		case 3: // Kepler
			cores = mp * 192;
			break;
		case 5: // Maxwell
			cores = mp * 128;
			break;
		case 6: // Pascal
			if (devProp.minor >= 1) cores = mp * 128;
			else if (devProp.minor == 0) cores = mp * 64;
			else log_inf << "Unknown device type!" << log_endl; 
			break;
		case 7: // Volta
			if (devProp.minor == 0) cores = mp * 64;
			else log_inf << "Unknown device type!" << log_endl;
			break;
		default:
			log_inf << "Unknown device type!" << log_endl; 
			break;
	}
    return cores;
}
