#pragma once

#include<CL/cl.h>

extern cl_context globalContext;
extern cl_device_id globalDevice;
extern cl_command_queue globalQueue;

void checkError(int value);

void buildIfNeeded(cl_program *program, cl_kernel *kernel, const char *kernel_name,
                   const char **srcStr, const size_t *srcLen);

void initCL(cl_device_id device, cl_context context);
void initCL_nvidia();
void freeCL();
