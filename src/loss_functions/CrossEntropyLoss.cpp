/**
 * @file
 * @brief implementacija CrossEntropyLoss funkcije
 * @author Jakov Novak
 */

#include "loss_functions/CrossEntropyLoss.h"
#include "Util.h"
#include "autograd_core/expression.hpp"

#include <CL/cl.h>
#include <cstring>

CrossEntropyLoss::CrossEntropyLoss(std::shared_ptr<autograd::Expression<Matrix>> left, std::shared_ptr<autograd::Expression<Matrix>> right) :
  BinaryOperator(left, right), program{ nullptr }, loss_kernel { nullptr }, grad_kernel { nullptr } { }

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

void CrossEntropyLoss::eval() {
  buildIfNeeded(&program, &loss_kernel, "CrossEntropyLoss", code,  lengths);

  Matrix input = this->left->getValue();
  Matrix expected = this->right->getValue();

  int _err;
  cl_mem output_buffer = clCreateBuffer(globalContext, CL_MEM_READ_ONLY, input.getN() * sizeof(float), nullptr, &_err);
  checkError(_err);

  checkError(clSetKernelArg(this->loss_kernel, 0, sizeof(float *), &input.data->data));
  checkError(clSetKernelArg(this->loss_kernel, 1, sizeof(float *), &expected.data->data));
  checkError(clSetKernelArg(this->loss_kernel, 2, sizeof(float *), &output_buffer));
  const size_t M = input.getM();
  checkError(clSetKernelArg(this->loss_kernel, 3, sizeof(const int), &M));

  const size_t global_work_size[] = { input.getN() };
  checkError(clEnqueueNDRangeKernel(globalQueue, this->loss_kernel, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  this->value = Matrix(output_buffer, input.getN(), 1);
  return;
}

// TODO: slozi ovo
void _derive(std::shared_ptr<autograd::Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<autograd::Expression<Matrix>>> &out_map) {
  return;
}
