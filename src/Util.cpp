/**
 * @file
 * @brief
 * @author Jakov Novak
 */

#include "Util.h"
#include <CL/cl.h>
#include <clblast.h>
#include <cstring>
#include <stdexcept>

cl_context globalContext;
cl_device_id globalDevice;
cl_command_queue globalQueue;

void checkError(int value) {
  switch (value) {
    case 0:
      // no error - continue!
      return;

    case -1: throw std::runtime_error("CL_DEVICE_NOT_FOUND");
    case -2: throw std::runtime_error("CL_DEVICE_NOT_AVAILABLE");
    case -3: throw std::runtime_error("CL_COMPILER_NOT_AVAILABLE");
    case -4: throw std::runtime_error("CL_MEM_OBJECT_ALLOCATION_FAILURE");
    case -5: throw std::runtime_error("CL_OUT_OF_RESOURCES");
    case -6: throw std::runtime_error("CL_OUT_OF_HOST_MEMORY");
    case -7: throw std::runtime_error("CL_PROFILING_INFO_NOT_AVAILABLE");
    case -8: throw std::runtime_error("CL_MEM_COPY_OVERLAP");
    case -9: throw std::runtime_error("CL_IMAGE_FORMAT_MISMATCH");
    case -10: throw std::runtime_error("CL_IMAGE_FORMAT_NOT_SUPPORTED");
    case -11: throw std::runtime_error("CL_BUILD_PROGRAM_FAILURE");
    case -12: throw std::runtime_error("CL_MAP_FAILURE");
    case -13: throw std::runtime_error("CL_MISALIGNED_SUB_BUFFER_OFFSET");
    case -14: throw std::runtime_error("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST");
    case -15: throw std::runtime_error("CL_COMPILE_PROGRAM_FAILURE");
    case -16: throw std::runtime_error("CL_LINKER_NOT_AVAILABLE");
    case -17: throw std::runtime_error("CL_LINK_PROGRAM_FAILURE");
    case -18: throw std::runtime_error("CL_DEVICE_PARTITION_FAILED");
    case -19: throw std::runtime_error("CL_KERNEL_ARG_INFO_NOT_AVAILABLE");
    // compile-time error)s
    case -30: throw std::runtime_error("CL_INVALID_VALUE");
    case -31: throw std::runtime_error("CL_INVALID_DEVICE_TYPE");
    case -32: throw std::runtime_error("CL_INVALID_PLATFORM");
    case -33: throw std::runtime_error("CL_INVALID_DEVICE");
    case -34: throw std::runtime_error("CL_INVALID_CONTEXT");
    case -35: throw std::runtime_error("CL_INVALID_QUEUE_PROPERTIES");
    case -36: throw std::runtime_error("CL_INVALID_COMMAND_QUEUE");
    case -37: throw std::runtime_error("CL_INVALID_HOST_PTR");
    case -38: throw std::runtime_error("CL_INVALID_MEM_OBJECT");
    case -39: throw std::runtime_error("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR");
    case -40: throw std::runtime_error("CL_INVALID_IMAGE_SIZE");
    case -41: throw std::runtime_error("CL_INVALID_SAMPLER");
    case -42: throw std::runtime_error("CL_INVALID_BINARY");
    case -43: throw std::runtime_error("CL_INVALID_BUILD_OPTIONS");
    case -44: throw std::runtime_error("CL_INVALID_PROGRAM");
    case -45: throw std::runtime_error("CL_INVALID_PROGRAM_EXECUTABLE");
    case -46: throw std::runtime_error("CL_INVALID_KERNEL_NAME");
    case -47: throw std::runtime_error("CL_INVALID_KERNEL_DEFINITION");
    case -48: throw std::runtime_error("CL_INVALID_KERNEL");
    case -49: throw std::runtime_error("CL_INVALID_ARG_INDEX");
    case -50: throw std::runtime_error("CL_INVALID_ARG_VALUE");
    case -51: throw std::runtime_error("CL_INVALID_ARG_SIZE");
    case -52: throw std::runtime_error("CL_INVALID_KERNEL_ARGS");
    case -53: throw std::runtime_error("CL_INVALID_WORK_DIMENSION");
    case -54: throw std::runtime_error("CL_INVALID_WORK_GROUP_SIZE");
    case -55: throw std::runtime_error("CL_INVALID_WORK_ITEM_SIZE");
    case -56: throw std::runtime_error("CL_INVALID_GLOBAL_OFFSET");
    case -57: throw std::runtime_error("CL_INVALID_EVENT_WAIT_LIST");
    case -58: throw std::runtime_error("CL_INVALID_EVENT");
    case -59: throw std::runtime_error("CL_INVALID_OPERATION");
    case -60: throw std::runtime_error("CL_INVALID_GL_OBJECT");
    case -61: throw std::runtime_error("CL_INVALID_BUFFER_SIZE");
    case -62: throw std::runtime_error("CL_INVALID_MIP_LEVEL");
    case -63: throw std::runtime_error("CL_INVALID_GLOBAL_WORK_SIZE");
    case -64: throw std::runtime_error("CL_INVALID_PROPERTY");
    case -65: throw std::runtime_error("CL_INVALID_IMAGE_DESCRIPTOR");
    case -66: throw std::runtime_error("CL_INVALID_COMPILER_OPTIONS");
    case -67: throw std::runtime_error("CL_INVALID_LINKER_OPTIONS");
    case -68: throw std::runtime_error("CL_INVALID_DEVICE_PARTITION_COUNT");

    // extension errors
    case -1000: throw std::runtime_error("CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR");
    case -1001: throw std::runtime_error("CL_PLATFORM_NOT_FOUND_KHR");
    case -1002: throw std::runtime_error("CL_INVALID_D3D10_DEVICE_KHR");
    case -1003: throw std::runtime_error("CL_INVALID_D3D10_RESOURCE_KHR");
    case -1004: throw std::runtime_error("CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR");
    case -1005: throw std::runtime_error("CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR");
    default: throw std::runtime_error("Unknown OpenCL error");
  }
}

void buildIfNeeded(cl_program *program, cl_kernel *kernel, const char *kernel_name,
                   const char **srcStr, const size_t *srcLen) {
  int _err;
  if(*program == nullptr) {
    *program = clCreateProgramWithSource(globalContext, 1, srcStr, srcLen, &_err);
    checkError(_err);
    clBuildProgram(*program, 1, &globalDevice, nullptr, nullptr, nullptr);
  }
  if(*kernel == nullptr) {
    *kernel = clCreateKernel(*program, kernel_name, &_err);
    checkError(_err);
  }
}

void initCL_nvidia() {
  cl_device_id devices[2];   
  cl_platform_id platforms[2];
  cl_uint num_devices;
  cl_uint num_platforms;
  checkError(clGetPlatformIDs(2, platforms, &num_platforms));

  for(cl_uint i=0;i < num_platforms;i++) {
    checkError(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 1,
            devices, &num_devices));
    char buffer[1024];

    for(cl_uint j=0;j < num_devices;j++) {
      checkError(clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
      // nasli smo dobar uredaj
      if(!strcmp("NVIDIA Corporation", buffer)) {
        checkError(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
        globalDevice = devices[j];
      }
    }
  }
  
  cl_int _err;
  globalContext = clCreateContext(NULL, 1, &globalDevice, nullptr, NULL, &_err);
  checkError(_err);

  initCL(globalDevice, globalContext);
}

void initCL(cl_device_id device, cl_context context) {
  globalDevice = device;
  globalContext = context;

  int _err;
  globalQueue = clCreateCommandQueueWithProperties(context, device, nullptr, &_err);
  checkError(_err);
}

void freeCL() {
  checkError(clReleaseCommandQueue(globalQueue));
  checkError(clReleaseDevice(globalDevice));
  checkError(clReleaseContext(globalContext));
}

// Utility function to convert CLBlast status code to string
const char* StatusCodeToString(clblast::StatusCode status) {
    switch (status) {
        case clblast::StatusCode::kSuccess: return "Success";
        case clblast::StatusCode::kInvalidValue: return "Invalid value";
        case clblast::StatusCode::kInvalidCommandQueue: return "Invalid command queue";
        case clblast::StatusCode::kInvalidMemObject: return "Invalid memory object";
        case clblast::StatusCode::kInvalidKernel: return "Invalid kernel";
        case clblast::StatusCode::kInvalidArgIndex: return "Invalid argument index";
        case clblast::StatusCode::kInvalidArgValue: return "Invalid argument value";
        case clblast::StatusCode::kInvalidArgSize: return "Invalid argument size";
        case clblast::StatusCode::kInvalidBinary: return "Invalid binary";
        case clblast::StatusCode::kInvalidBuildOptions: return "Invalid build options";
        case clblast::StatusCode::kInvalidProgram: return "Invalid program";
        case clblast::StatusCode::kInvalidProgramExecutable: return "Invalid program executable";
        case clblast::StatusCode::kInvalidKernelName: return "Invalid kernel name";
        case clblast::StatusCode::kInvalidKernelDefinition: return "Invalid kernel definition";
        case clblast::StatusCode::kInvalidGlobalOffset: return "Invalid global offset";
        case clblast::StatusCode::kInvalidEventWaitList: return "Invalid event wait list";
        case clblast::StatusCode::kInvalidEvent: return "Invalid event";
        case clblast::StatusCode::kInvalidOperation: return "Invalid operation";
        case clblast::StatusCode::kInvalidBufferSize: return "Invalid buffer size";
        case clblast::StatusCode::kInvalidGlobalWorkSize: return "Invalid global work size";
        default: return "Unknown error";
    }
}
