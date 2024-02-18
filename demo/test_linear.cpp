#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "../src/layers/Layers.h"
#include "../src/loss_functions/LossFunctions.h"
#include "../src/Network.h"
#include "../src/Util.h"
#include "../src/optimizers/Optimizers.h"

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
  std::cerr << "OpenCL Error (via pfn_notify): " << errinfo << std::endl;
  throw std::runtime_error("OpenCL");
}

int main() {
  cl_device_id devices[2];   
  cl_platform_id platforms[2];
  cl_device_id device_id;
  cl_uint num_devices;
  cl_uint num_platforms;
  checkError(clGetPlatformIDs(2, platforms, &num_platforms));

  for(int i=0;i < num_platforms;i++) {
    checkError(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 1,
            devices, &num_devices));
    char buffer[1024];

    for(int j=0;j < num_devices;j++) {
      checkError(clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
      printf("Pronaden uredaj: %s\n", buffer);
      // nasli smo dobar uredaj
      if(!strcmp("NVIDIA Corporation", buffer)) {
        checkError(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
        printf("Koristim gpu: %s\n", buffer);
        device_id = devices[j];
      }
    }
  }
  
  cl_int _err;
  cl_context context = clCreateContext(NULL, 1, &device_id, pfn_notify, NULL, &_err);
  checkError(_err);

  Network::ILayer *sloj1 = new Linear(context, 2, 2);
  Network::ILayer *sloj2 = new Linear(context, 2, 3);
  Network::ILayer *sloj3 = new Sigmoid();
  Network mreza(context, device_id, { sloj1, sloj2, sloj3 });

  float host_ptr[] = { 2, 1.0,
                      2.5, 10.2,
                      1.5, 1.7,
                      1.8, 3.8,
                      4.8, 6.4 };
  cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 5*2, host_ptr, &_err);
  checkError(_err);

  Matrix izlaz = mreza.forward(input_buffer, 5, 2);
  cl_command_queue kju = clCreateCommandQueueWithProperties(context, device_id, nullptr, &_err);
  checkError(_err);

  cl_event user_event = clCreateUserEvent(context, &_err);
  checkError(_err);
  float *dobiveno = new float[5*3];
  checkError(clEnqueueReadBuffer(kju, izlaz.data, CL_TRUE, 0, 5*3*sizeof(float), dobiveno, 0, nullptr, &user_event));
  clWaitForEvents(1, &user_event);
  clReleaseEvent(user_event);

  // ispis rezultata
  std::cout << "host_ptr nakon dva potpuno povezana sloja:" << std::endl;
  float *ocekivano = new float[5*3];
  for(int i=0;i < 5;i++) {
    for(int j=0;j < 3;j++) {
      std::cout << dobiveno[i*3 + j] << ' ';
      ocekivano[i*3 + j] = float(rand() % 2) + dobiveno[i*3 + 1];
    }
    std::cout << std::endl;
  }
  cl_mem ocekivano_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 5*3*sizeof(float), ocekivano, &_err);
  checkError(_err);

  delete[] dobiveno;
  delete[] ocekivano;

  Matrix probs = { izlaz.data, 5, 3 };
  Matrix expected = { ocekivano_cl, 5, 3 };
  ILossFunc *loss_func = new CrossEntropyLoss();
  mreza.backward(probs, expected, loss_func, new SGD(0.05f));

  // for(int i=0;i < 100000;i++) {
  //   input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 5*2*sizeof(float), host_ptr, &_err);
  //   checkError(_err);
  //   Matrix izlaz = mreza.forward({ input_buffer, 5, 2 });
  //   mreza.backward(izlaz, expected, loss_func, nullptr);
  // }

  clReleaseCommandQueue(kju);
  clReleaseDevice(device_id);
  clReleaseContext(context);
  return 0;
}
