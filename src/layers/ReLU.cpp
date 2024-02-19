#include "ReLU.h"
#include "../Util.h"
#include <CL/cl.h>
#include <cstring>

#ifdef DEBUG
#include <iostream>
#endif

ReLU::ReLU() : program{ nullptr }, forward_kernel{ nullptr }, backward_kernel{ nullptr }, last_output{ nullptr } { }

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

std::shared_ptr<Matrix> ReLU::forward(Network &network, std::shared_ptr<Matrix> input_matrix) {
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

  cl_mem output_buffer = clCreateBuffer(getContext(network), CL_MEM_READ_ONLY, input_matrix->N * input_matrix->M * sizeof(float), nullptr, &_err);
  checkError(_err);

  checkError(clSetKernelArg(this->forward_kernel, 0, sizeof(float *), &input_matrix->data));
  checkError(clSetKernelArg(this->forward_kernel, 1, sizeof(float *), &output_buffer));
  checkError(clSetKernelArg(this->forward_kernel, 2, sizeof(const int), &input_matrix->M));

  const size_t global_work_size[] = { input_matrix->N, input_matrix->M };
  checkError(_err);
  checkError(clEnqueueNDRangeKernel(getQueue(network), this->forward_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));
  
  this->last_output = std::make_shared<Matrix>(output_buffer, input_matrix->N, input_matrix->M);
  return this->last_output;
}

std::shared_ptr<Matrix> ReLU::backward(Network &network, std::shared_ptr<Matrix> output_grad, std::weak_ptr<IOptimizer> optim) {
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

  cl_mem output_buffer = clCreateBuffer(getContext(network), CL_MEM_READ_ONLY, output_grad->N * output_grad->M * sizeof(float), nullptr, &_err);
  checkError(_err);

  checkError(clSetKernelArg(this->backward_kernel, 0, sizeof(float *), &output_grad->data));
  checkError(clSetKernelArg(this->backward_kernel, 1, sizeof(float *), &this->last_output->data));
  checkError(clSetKernelArg(this->backward_kernel, 2, sizeof(float *), &output_buffer));
  checkError(clSetKernelArg(this->backward_kernel, 3, sizeof(const int), &output_grad->M));

  const size_t global_work_size[] = { output_grad->N, output_grad->M };
  checkError(_err);
  checkError(clEnqueueNDRangeKernel(getQueue(network), this->backward_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  return std::make_shared<Matrix>(output_buffer, output_grad->N, output_grad->M);
}
