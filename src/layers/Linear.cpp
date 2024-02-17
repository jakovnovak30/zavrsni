#include "Linear.h"
#include "../Network.h"
#include "../Util.h"

#include <CL/cl.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdexcept>

#ifdef DEBUG
#include <iostream>
#endif

inline float gen_uniform(float k) {
  static bool called = false;
  if(!called) {
    srand((unsigned) time(nullptr));
    called = true;
  }

  float HI = sqrt(k);
  float LO = -HI;
  return LO + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/(HI-LO)));
}

Linear::Linear(cl_context context, const size_t in_features, const size_t out_features) : Linear(context, in_features, out_features, true) { }

// initialize values from U(-sqrt(k), +sqrt(k)), where k = 1 / in_features
Linear::Linear(cl_context context, const size_t in_features, const size_t out_features, bool bias) : in_features{ in_features }, out_features{ out_features }
{
  int _err;
  float *host_ptr = new float[in_features*out_features];
  for(size_t i=0;i < in_features*out_features;i++) {
    host_ptr[i] = gen_uniform(1.0f / in_features);
  }
  this->parameters = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, in_features * out_features * sizeof(float), host_ptr, &_err);
  checkError(_err);
  delete[] host_ptr;

  if(bias) {
    float *host_ptr = new float[out_features];
    for(int i=0;i < out_features;i++)
      host_ptr[i] = gen_uniform(1.0f / in_features);
    this->biases = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, out_features * sizeof(float), host_ptr, &_err);
    checkError(_err);
    delete[] host_ptr;
  }
  else {
    this->biases = nullptr;
  }

  // kompajlira se u trenutku poziva
  this->program = nullptr;
  this->forward_kernel = nullptr;
  this->backward_kernel = nullptr;
  this->bias_kernel = nullptr;
}

Linear::~Linear() {
  #ifdef DEBUG
  std::cout << "[DEBUG]: Pozvan destruktor za Linear!" << std::endl;
  #endif

  checkError(clReleaseMemObject(this->parameters));

  if(this->biases != nullptr)
    checkError(clReleaseMemObject(this->biases));
  if(this->forward_kernel != nullptr)
    checkError(clReleaseKernel(this->forward_kernel));
  if(this->backward_kernel != nullptr)
    checkError(clReleaseKernel(this->backward_kernel));
  if(this->bias_kernel != nullptr)
    checkError(clReleaseKernel(this->bias_kernel));
  if(this->program != nullptr)
    checkError(clReleaseProgram(this->program));
}

static const char *code[] = 
            {
              #include "../kernels/Linear.cl"
            };
static const size_t lengths[] = { strlen(code[0]) };

Matrix Linear::forward(Network &network, Matrix &input_matrix) {
  // provjeri dal je valjan ulazni oblik
  if(input_matrix.M != this->in_features) {
    throw std::logic_error("Ne valja oblik ulaza u potpuno povezani sloj!");
  }
  int _err;

  if(this->program == nullptr) {
    this->program = clCreateProgramWithSource(getContext(network), 1, code, lengths, &_err);
    checkError(_err);
    cl_device_id device = getDevice(network);
    checkError(clBuildProgram(this->program, 1, &device, nullptr, nullptr, nullptr));
  }
  if(this->forward_kernel == nullptr) {
    this->forward_kernel = clCreateKernel(this->program, "matrixMultiplyTransposed", &_err);
    checkError(_err);
  }

  cl_mem input_buffer = input_matrix.data;
  checkError(clSetKernelArg(this->forward_kernel, 0, sizeof(float *), &input_buffer));
  checkError(clSetKernelArg(this->forward_kernel, 1, sizeof(float *), &this->parameters));
  // izlaz je oblika NxOut
  cl_mem output_buffer = clCreateBuffer(getContext(network), CL_MEM_READ_WRITE, input_matrix.N * out_features * sizeof(float), nullptr, &_err);
  checkError(_err);
  checkError(clSetKernelArg(this->forward_kernel, 2, sizeof(float *), &output_buffer));

  const size_t N = input_matrix.N;
  checkError(clSetKernelArg(this->forward_kernel, 3, sizeof(const int), &N));
  checkError(clSetKernelArg(this->forward_kernel, 4, sizeof(const int), &this->in_features));
  checkError(clSetKernelArg(this->forward_kernel, 5, sizeof(const int), &this->out_features));

  cl_event finish_event = clCreateUserEvent(getContext(network), &_err);
  checkError(_err);
  size_t global_work_size[] = { N, this->out_features };
  checkError(clEnqueueNDRangeKernel(getQueue(network), this->forward_kernel, 2, nullptr, &global_work_size[0], nullptr, 0, nullptr, &finish_event));
  checkError(clWaitForEvents(1, &finish_event));

  if(this->biases != nullptr) {
    if(this->bias_kernel == nullptr) {
      this->bias_kernel = clCreateKernel(this->program, "addBias", &_err);
      checkError(_err);
    }

    checkError(clSetKernelArg(this->bias_kernel, 0, sizeof(float *), &output_buffer));
    checkError(clSetKernelArg(this->bias_kernel, 1, sizeof(float *), &this->biases));
    checkError(clSetKernelArg(this->bias_kernel, 2, sizeof(const int), &this->out_features));
    size_t global_work_size[] = { N, this->out_features };
    checkError(clEnqueueNDRangeKernel(getQueue(network), this->bias_kernel, 2, nullptr, &global_work_size[0], nullptr, 0, nullptr, &finish_event));
    checkError(clWaitForEvents(1, &finish_event));
  }

  checkError(clReleaseEvent(finish_event));
  return { output_buffer, N, this->out_features };
}

Matrix Linear::backward(Network &network, Matrix &output_grad) {
  // TODO
  return {};
}
