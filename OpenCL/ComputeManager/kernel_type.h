#ifndef CL_KERNEL_TYPE
#define CL_KERNEL_TYPE

#include "CL/cl.h"

struct cl_kernel_t
{
    cl_program program;
    cl_kernel kernel;
    const char *text;
    size_t size;
};

#endif // CL_KERNEL_TYPE
