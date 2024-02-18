#include "ReLU.h"
#include "../Util.h"
#include <CL/cl.h>
#include <cstring>

#ifdef DEBUG
#include <iostream>
#endif

ReLU::ReLU() : program{ nullptr }, forward_kernel{ nullptr }, backward_kernel{ nullptr } {
  this->last_output = { nullptr, 0, 0 };
}

ReLU::~ReLU() {
  if(this->forward_kernel)
    checkError(clReleaseKernel(this->forward_kernel));
  if(this->backward_kernel)
    checkError(clReleaseKernel(this->backward_kernel));
  if(this->program != nullptr)
    checkError(clReleaseProgram(this->program));
}

static const char *code[] = 
                  {
                    #include "../kernels/ReLU.cl"
                  };
static const size_t lengths[] = { strlen(code[0]) };

Matrix ReLU::forward(Network &network, Matrix &input_matrix) {
  int _err;

  if(this->program == nullptr) {
    this->program = clCreateProgramWithSource(getContext(network), 1, code, lengths, &_err);
    checkError(_err);
    cl_device_id device = getDevice(network);
    checkError(clBuildProgram(this->program, 1, &device, nullptr, nullptr, nullptr));
  }
  if(this->forward_kernel == nullptr) {
    this->forward_kernel = clCreateKernel(this->program, "reluForward", &_err);
    checkError(_err);
  }

  if(this->last_output.data != nullptr) {
    checkError(clReleaseMemObject(this->last_output.data));
  }
  this->last_output.data = clCreateBuffer(getContext(network), CL_MEM_READ_ONLY, input_matrix.N * input_matrix.M * sizeof(float), nullptr, &_err);
  this->last_output.N = input_matrix.N;
  this->last_output.M = input_matrix.M;
  checkError(_err);

  checkError(clSetKernelArg(this->forward_kernel, 0, sizeof(float *), &input_matrix.data));
  checkError(clSetKernelArg(this->forward_kernel, 1, sizeof(float *), &this->last_output.data));
  checkError(clSetKernelArg(this->forward_kernel, 2, sizeof(const int), &input_matrix.M));

  const size_t global_work_size[] = { input_matrix.N, input_matrix.M };
  checkError(_err);
  checkError(clEnqueueNDRangeKernel(getQueue(network), this->forward_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));
  
  return this->last_output;
}

Matrix ReLU::backward(Network &network, Matrix &output_grad, IOptimizer *optim) {
  int _err;

  #ifdef DEBUG
  std::cout << "[DEBUG]: radim backprop na ReLU sloju!" << std::endl;
  #endif

  if(this->program == nullptr) {
    this->program = clCreateProgramWithSource(getContext(network), 1, code, lengths, &_err);
    checkError(_err);
    cl_device_id device = getDevice(network);
    checkError(clBuildProgram(this->program, 1, &device, nullptr, nullptr, nullptr));
  }
  if(this->backward_kernel == nullptr) {
    this->backward_kernel = clCreateKernel(this->program, "reluBackward", &_err);
    checkError(_err);
  }

  cl_mem output_buffer = clCreateBuffer(getContext(network), CL_MEM_READ_ONLY, output_grad.N * output_grad.M * sizeof(float), nullptr, &_err);
  checkError(_err);

  checkError(clSetKernelArg(this->backward_kernel, 0, sizeof(float *), &output_grad.data));
  checkError(clSetKernelArg(this->backward_kernel, 1, sizeof(float *), &this->last_output.data));
  checkError(clSetKernelArg(this->backward_kernel, 2, sizeof(float *), &output_buffer));
  checkError(clSetKernelArg(this->backward_kernel, 3, sizeof(const int), &output_grad.M));

  const size_t global_work_size[] = { output_grad.N, output_grad.M };
  checkError(_err);
  checkError(clEnqueueNDRangeKernel(getQueue(network), this->backward_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  return { output_buffer, output_grad.N, output_grad.M };
}
