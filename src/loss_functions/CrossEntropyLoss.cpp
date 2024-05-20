/**
 * @file
 * @brief implementacija CrossEntropyLossWithSoftmaxWithSoftmax funkcije
 * @author Jakov Novak
 */

#include "loss_functions/CrossEntropyLossWithSoftmax.h"
#include "loss_functions/Softmax.h"
#include "Util.h"
#include "autograd_core/expression.hpp"

#include <CL/cl.h>
#include <cstring>
#include <memory>

CrossEntropyLossWithSoftmax::CrossEntropyLossWithSoftmax(std::shared_ptr<autograd::Expression<Matrix>> left, std::shared_ptr<autograd::Expression<Matrix>> right) :
  BinaryOperator(left, right), program{ nullptr }, loss_kernel { nullptr } { }

CrossEntropyLossWithSoftmax::~CrossEntropyLossWithSoftmax() {
  if(this->loss_kernel != nullptr)
    checkError(clReleaseKernel(this->loss_kernel));
  if(this->program != nullptr)
    checkError(clReleaseProgram(this->program));
}

static const char *code[] =
                        {
                          #include "kernels/CrossEntropyLoss.cl"
                        };
static const size_t lengths[] = { strlen(code[0]) };

void CrossEntropyLossWithSoftmax::eval() {
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
void CrossEntropyLossWithSoftmax::_derive(std::shared_ptr<autograd::Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<autograd::Expression<Matrix>>> &out_map) {
  // Calculate the softmax of the input
  std::shared_ptr<autograd::Expression<Matrix>> softmax = std::make_shared<Softmax>(this->left);

  // Subtract the expected values from the softmax values to get the gradient of the loss
  auto grad_input = softmax - this->right;

  // Multiply by the seed if it is not nullptr
  if (seed) {
      grad_input = grad_input * seed;
  }

  this->left->derive(grad_input, out_map);
}

void CrossEntropyLossWithSoftmax::addSubgraph(Agraph_t *graph, Agnode_t *prev) const {
  static int id_counter = 1000;
  Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
  agset(curr, (char *) "label", "CrossEntropyLossWithSoftmax");
  agedge(graph, curr, prev, nullptr, 1);

  this->left->addSubgraph(graph, curr);
  this->right->addSubgraph(graph, curr);
}
