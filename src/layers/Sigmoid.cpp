#include "Sigmoid.h"
#include "../Util.h"
#include <CL/cl.h>
#include <cstring>
#include <stdexcept>

#ifdef DEBUG
#include <iostream>
#endif

Sigmoid::Sigmoid() : program{ nullptr }, forward_kernel{ nullptr }, backward_kernel{ nullptr } {
  this->last_output = { nullptr, 0, 0 };
}

Sigmoid::~Sigmoid() {
  #ifdef DEBUG
  std::cout << "[DEBUG]: pozvan destruktor za Sigmoid!" << std::endl;
  #endif
  if(this->forward_kernel != nullptr)
    checkError(clReleaseKernel(this->forward_kernel));
  if(this->backward_kernel != nullptr)
    checkError(clReleaseKernel(this->backward_kernel));
  if(this->program != nullptr)
    checkError(clReleaseProgram(this->program));
}

static const char *code[] = 
            {
              #include "../kernels/Sigmoid.cl"
            };
static const size_t lengths[] = { strlen(code[0]) };

Matrix Sigmoid::forward(Network &network, Matrix &input_matrix) {
  int _err;
  if(this->program == nullptr) {
    this->program = clCreateProgramWithSource(getContext(network), 1, code, lengths, &_err);
    checkError(_err);
    cl_device_id device = getDevice(network);
    checkError(clBuildProgram(this->program, 1, &device, nullptr, nullptr, nullptr));
  }
  if(this->forward_kernel == nullptr) {
    this->forward_kernel = clCreateKernel(this->program, "sigmoidForward", &_err);
    checkError(_err);
  }

  if(this->last_output.data != nullptr) {
    checkError(clReleaseMemObject(this->last_output.data));
  }

  this->last_output.data = clCreateBuffer(getContext(network), CL_MEM_READ_WRITE, input_matrix.N * input_matrix.M * sizeof(float), nullptr, &_err);
  this->last_output.N = input_matrix.N;
  this->last_output.M = input_matrix.M;
  checkError(_err);

  checkError(clSetKernelArg(this->forward_kernel, 0, sizeof(float *), &input_matrix.data));
  checkError(clSetKernelArg(this->forward_kernel, 1, sizeof(float *), &this->last_output.data));
  checkError(clSetKernelArg(this->forward_kernel, 2, sizeof(const int), &input_matrix.M));

  const size_t global_work_size[] = { input_matrix.N, input_matrix.M };
  cl_event user_event = clCreateUserEvent(getContext(network), &_err);
  checkError(_err);
  checkError(clEnqueueNDRangeKernel(getQueue(network), this->forward_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, &user_event));
  checkError(clWaitForEvents(1, &user_event));
  checkError(clReleaseEvent(user_event));

  return this->last_output;
}

Matrix Sigmoid::backward(Network &network, Matrix &output_grad) {
  if(output_grad.N != this->last_output.N || output_grad.M != this->last_output.M)
    throw std::logic_error("Dimenzije matrica ne odgovaraju!");

  int _err;
  if(this->program == nullptr) {
    this->program = clCreateProgramWithSource(getContext(network), 1, code, lengths, &_err);
    checkError(_err);
    cl_device_id device = getDevice(network);
    checkError(clBuildProgram(this->program, 1, &device, nullptr, nullptr, nullptr));
  }
  if(this->backward_kernel == nullptr) {
    this->backward_kernel = clCreateKernel(this->program, "sigmoidBackward", &_err);
    checkError(_err);
  }

  cl_mem output_buffer = clCreateBuffer(getContext(network), CL_MEM_READ_ONLY, output_grad.N * output_grad.M * sizeof(float), nullptr, &_err);
  checkError(_err);

  checkError(clSetKernelArg(this->backward_kernel, 0, sizeof(float *), &output_grad.data));
  checkError(clSetKernelArg(this->backward_kernel, 1, sizeof(float *), &this->last_output.data));
  checkError(clSetKernelArg(this->backward_kernel, 2, sizeof(float *), &output_buffer));
  checkError(clSetKernelArg(this->backward_kernel, 3, sizeof(float *), &output_grad.M));

  const size_t global_work_size[] = { output_grad.N, output_grad.M };
  cl_event user_event = clCreateUserEvent(getContext(network), &_err);
  checkError(_err);
  checkError(clEnqueueNDRangeKernel(getQueue(network), this->backward_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, &user_event));
  checkError(clWaitForEvents(1, &user_event));
  checkError(clReleaseEvent(user_event));

  return { output_buffer, output_grad.N, output_grad.M };
}
