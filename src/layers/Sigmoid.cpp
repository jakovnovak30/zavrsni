/**
 * @file
 * @brief implementacije Sigmoid funkcije
 * @author Jakov Novak
 */

#include "layers/Sigmoid.h"
#include "Util.h"
#include "autograd_core/basic_operations.hpp"
#include "autograd_core/expression.hpp"

#include <CL/cl.h>
#include <cstring>
#include <graphviz/cgraph.h>

using autograd::UnaryOperator;

Sigmoid::Sigmoid(std::shared_ptr<Expression<Matrix>> prev)
    : UnaryOperator<Matrix>(prev), program{nullptr}, forward_kernel{nullptr} {}

Sigmoid::~Sigmoid() {
  if(this->forward_kernel != nullptr)
    checkError(clReleaseKernel(this->forward_kernel));
  if(this->program != nullptr)
    checkError(clReleaseProgram(this->program));
}

static const char *code[] = 
            {
              #include "kernels/Sigmoid.cl"
            };
static const size_t lengths[] = { strlen(code[0]) };

void Sigmoid::eval() {
  Matrix input_matrix = this->prev->getValue();
  buildIfNeeded(&program, &forward_kernel, "sigmoidForward", code, lengths);

  int _err;
  cl_mem output_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, input_matrix.getM() * input_matrix.getN() * sizeof(float), nullptr, &_err);
  checkError(_err);

  checkError(clSetKernelArg(this->forward_kernel, 0, sizeof(float *), &input_matrix.data->data));
  checkError(clSetKernelArg(this->forward_kernel, 1, sizeof(float *), &output_buffer));
  const size_t M = input_matrix.getM();
  checkError(clSetKernelArg(this->forward_kernel, 2, sizeof(const int), &M));

  const size_t global_work_size[] = { input_matrix.getN(), input_matrix.getM() };
  checkError(_err);
  checkError(clEnqueueNDRangeKernel(globalQueue, this->forward_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  this->value = Matrix(output_buffer, input_matrix.getN(), input_matrix.getM());
  return;
}

// f(x) * (1 - f(x)) je derivacija
void Sigmoid::_derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) {
  using namespace autograd;
  auto f_x = this->shared_from_this();
  this->prev->derive(std::make_shared<Mult<Matrix>>(std::make_shared<Sub<Matrix>>(f_x, std::make_shared<Mult<Matrix>>(f_x, f_x)), seed), out_map);
}

void Sigmoid::addSubgraph(Agraph_t *graph, Agnode_t *prev) const {
  using namespace autograd;
  Agnode_t *curr = agnode(graph, (char *) (std::string("sigmoid") + std::to_string(id_counter++)).c_str(), 1);
  agset(curr, (char *) "label", "sigmoid");
  agedge(graph, curr, prev, nullptr, 1);

  this->prev->addSubgraph(graph, curr);
}
