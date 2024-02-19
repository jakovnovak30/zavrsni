#include "SGD.h"
#include "../Util.h"
#include "../Network.h"
#include <CL/cl.h>
#include <cstring>
#include <stdexcept>

SGD::SGD(float learning_rate) : learning_rate{ learning_rate }, program{ nullptr }, optimize_kernel{ nullptr } { }

SGD::~SGD() {
  if(this->optimize_kernel != nullptr)
    checkError(clReleaseKernel(this->optimize_kernel));
  if(this->program != nullptr)
    checkError(clReleaseProgram(this->program));
}

static const char *code[] =
                        {
                          #include "../kernels/SGD.cl"
                        };
static size_t lengths[] = { strlen(code[0]) };

void SGD::optimize(Network &network, std::shared_ptr<Matrix> parameters, std::shared_ptr<Matrix> gradients) {
  if(parameters->N != gradients->N || parameters->M != gradients->M)
    throw std::logic_error("Krive dimenzije matrica parametara i gradijenata!");

  int _err;
  if(this->program == nullptr) {
    this->program = clCreateProgramWithSource(getContext(network), 1, code, lengths, &_err);
    checkError(_err);
    cl_device_id device = getDevice(network);
    checkError(clBuildProgram(this->program, 1, &device, nullptr, nullptr, nullptr));
  }
  if(this->optimize_kernel == nullptr) {
    this->optimize_kernel = clCreateKernel(this->program, "optimizationStep", &_err);
    checkError(_err);
  }

  checkError(clSetKernelArg(this->optimize_kernel, 0, sizeof(float *), &parameters->data));
  checkError(clSetKernelArg(this->optimize_kernel, 1, sizeof(float *), &gradients->data));
  checkError(clSetKernelArg(this->optimize_kernel, 2, sizeof(const float), &this->learning_rate));
  checkError(clSetKernelArg(this->optimize_kernel, 3, sizeof(const int), &parameters->M));

  const size_t global_work_size[] = { parameters->N, parameters->M };
  checkError(clEnqueueNDRangeKernel(getQueue(network), this->optimize_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  return;
}
