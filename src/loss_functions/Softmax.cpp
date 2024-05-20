#include "loss_functions/Softmax.h"
#include "Util.h"
#include "autograd_core/expression.hpp"
#include <CL/cl.h>
#include <cassert>
#include <stdexcept>

Softmax::Softmax(std::shared_ptr<Expression<Matrix>> prev) : UnaryOperator(prev), program(nullptr), eval_kernel(nullptr) {}

Softmax::~Softmax() {
  if(this->eval_kernel != nullptr)
    checkError(clReleaseKernel(this->eval_kernel));
  if(this->program != nullptr)
    checkError(clReleaseProgram(this->program));
}

static const char *code[] =
                        {
                          #include "kernels/Softmax.cl"
                        };
static const size_t lengths[] = { strlen(code[0]) };

void Softmax::eval() {
  buildIfNeeded(&this->program, &this->eval_kernel, "Softmax", code, lengths);

  Matrix input = this->prev->getValue();

  // vektor-stupac
  assert(input.getM() == 1);
  int _err;
  cl_mem out_buffer = clCreateBuffer(globalContext, CL_MEM_READ_ONLY, input.getN() * sizeof(float), nullptr, &_err);
  checkError(_err);

  const unsigned int N = input.getN();
  checkError(clSetKernelArg(this->eval_kernel, 0, sizeof(float *), &input.data->data));
  checkError(clSetKernelArg(this->eval_kernel, 1, sizeof(float *), &out_buffer));
  checkError(clSetKernelArg(this->eval_kernel, 2, sizeof(unsigned int), &N));

  const size_t global_work_size[] = { input.getN() };
  checkError(clEnqueueNDRangeKernel(globalQueue, this->eval_kernel, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  this->value = Matrix(out_buffer, input.getN(), 1);
  return;
}

void Softmax::_derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) {
  throw std::logic_error("not implemented yet!");
}
