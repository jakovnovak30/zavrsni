#include "CrossEntropyLoss.h"
#include "../Util.h"
#include <CL/cl.h>
#include <cstring>
#include <stdexcept>

CrossEntropyLoss::CrossEntropyLoss() : program{ nullptr }, loss_kernel { nullptr }, grad_kernel { nullptr } { }

CrossEntropyLoss::~CrossEntropyLoss() {
  if(this->loss_kernel != nullptr)
    checkError(clReleaseKernel(this->loss_kernel));
  if(this->grad_kernel != nullptr)
    checkError(clReleaseKernel(this->grad_kernel));
  if(this->program != nullptr)
    checkError(clReleaseProgram(this->program));
}

static const char *code[] =
                        {
                          #include "../kernels/CrossEntropyLoss.cl"
                        };
static const size_t lengths[] = { strlen(code[0]) };

std::shared_ptr<Matrix> CrossEntropyLoss::calculate_loss(Network &network, std::shared_ptr<Matrix> input, std::shared_ptr<Matrix> expected) {
  if(input->N != expected->N || input->M != expected->M)
    throw std::logic_error("Dimenzije matrica se ne poklapaju!");

  int _err;
  if(this->program == nullptr) {
    this->program = clCreateProgramWithSource(getContext(network), 1, code, lengths, &_err);
    checkError(_err);
    cl_device_id device = getDevice(network);
    checkError(clBuildProgram(this->program, 1, &device, nullptr, nullptr, nullptr));
  }
  if(this->loss_kernel == nullptr) {
    this->loss_kernel = clCreateKernel(this->program, "calculateLoss", &_err);
    checkError(_err);
  }

  cl_mem output_buffer = clCreateBuffer(getContext(network), CL_MEM_READ_ONLY, input->N * sizeof(float), nullptr, &_err);
  checkError(_err);

  checkError(clSetKernelArg(this->loss_kernel, 0, sizeof(float *), &input->data));
  checkError(clSetKernelArg(this->loss_kernel, 1, sizeof(float *), &expected->data));
  checkError(clSetKernelArg(this->loss_kernel, 2, sizeof(float *), &output_buffer));
  checkError(clSetKernelArg(this->loss_kernel, 3, sizeof(const int), &input->M));

  const size_t global_work_size[] = { input->N };
  checkError(clEnqueueNDRangeKernel(getQueue(network), this->loss_kernel, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  return std::make_shared<Matrix>(output_buffer, input->N, 1);
}

std::shared_ptr<Matrix> CrossEntropyLoss::calculate_gradient(Network &network, std::shared_ptr<Matrix> input, std::shared_ptr<Matrix> expected) {
  int _err;
  if(this->program == nullptr) {
    this->program = clCreateProgramWithSource(getContext(network), 1, code, lengths, &_err);
    checkError(_err);
    cl_device_id device = getDevice(network);
    checkError(clBuildProgram(this->program, 1, &device, nullptr, nullptr, nullptr));
  }
  if(this->loss_kernel == nullptr) {
    this->loss_kernel = clCreateKernel(this->program, "calculateGradient", &_err);
    checkError(_err);
  }

  // izlaz ima dimenzije NxC
  cl_mem output_buffer = clCreateBuffer(getContext(network), CL_MEM_READ_ONLY, input->N * input->M * sizeof(float), nullptr, &_err);
  checkError(_err);

  checkError(clSetKernelArg(this->loss_kernel, 0, sizeof(float *), &input->data));
  checkError(clSetKernelArg(this->loss_kernel, 1, sizeof(float *), &expected->data));
  checkError(clSetKernelArg(this->loss_kernel, 2, sizeof(float *), &output_buffer));
  checkError(clSetKernelArg(this->loss_kernel, 3, sizeof(const int), &input->M));

  const size_t global_work_size[] = { input->N, input->M };

  checkError(clEnqueueNDRangeKernel(getQueue(network), this->loss_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  return std::make_shared<Matrix>(output_buffer, input->N, input->M);
}
