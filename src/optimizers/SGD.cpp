/**
 * @file
 * @brief SGD implementacija
 * @author Jakov Novak
 */

#include "optimizers/SGD.h"
#include "Util.h"

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

void SGD::optimize(Matrix &parameters, Matrix &gradients) {
  if(parameters.getN() != gradients.getN() || parameters.getM() != gradients.getM())
    throw std::logic_error("Krive dimenzije matrica parametara i gradijenata!");

  int _err;
  if(this->program == nullptr) {
    this->program = clCreateProgramWithSource(globalContext, 1, code, lengths, &_err);
    checkError(_err);
    checkError(clBuildProgram(this->program, 1, &globalDevice, nullptr, nullptr, nullptr));
  }
  if(this->optimize_kernel == nullptr) {
    this->optimize_kernel = clCreateKernel(this->program, "optimizationStep", &_err);
    checkError(_err);
  }

  checkError(clSetKernelArg(this->optimize_kernel, 0, sizeof(float *), &parameters.data));
  checkError(clSetKernelArg(this->optimize_kernel, 1, sizeof(float *), &gradients.data));
  checkError(clSetKernelArg(this->optimize_kernel, 2, sizeof(const float), &this->learning_rate));
  const size_t M = parameters.getM();
  checkError(clSetKernelArg(this->optimize_kernel, 3, sizeof(const int), &M));

  const size_t global_work_size[] = { parameters.getN(), parameters.getM() };
  checkError(clEnqueueNDRangeKernel(globalQueue, this->optimize_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  return;
}
