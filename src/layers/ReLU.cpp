#include "layers/ReLU.h"
#include "Util.h"

#include <CL/cl.h>
#include <cstring>

ReLU::ReLU(std::shared_ptr<Expression<Matrix>> prev) : 
  autograd::UnaryOperator<Matrix>(prev), program{ nullptr }, forward_kernel{ nullptr } { }

ReLU::~ReLU() {
  if(this->forward_kernel)
    checkError(clReleaseKernel(this->forward_kernel));
  if(this->program != nullptr)
    checkError(clReleaseProgram(this->program));
}

static const char *code[] = 
                  {
                    #include "../kernels/ReLU.cl"
                  };
static const size_t lengths[] = { strlen(code[0]) };

void ReLU::eval() {
  buildIfNeeded(&program, &forward_kernel, "reluForward", code, lengths);

  Matrix input_matrix = this->prev->getValue();
  int _err;
  cl_mem output_buffer = clCreateBuffer(globalContext, CL_MEM_READ_ONLY, input_matrix.getN() * input_matrix.getM() * sizeof(float), nullptr, &_err);
  checkError(_err);

  checkError(clSetKernelArg(this->forward_kernel, 0, sizeof(float *), &input_matrix.data->data));
  checkError(clSetKernelArg(this->forward_kernel, 1, sizeof(float *), &output_buffer));
  const int M = input_matrix.getM();
  checkError(clSetKernelArg(this->forward_kernel, 2, sizeof(const int), &M));

  const size_t global_work_size[] = { input_matrix.getN(), input_matrix.getM() };
  checkError(_err);
  checkError(clEnqueueNDRangeKernel(globalQueue, this->forward_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));
  
  this->value = Matrix(output_buffer, input_matrix.getN(), input_matrix.getM());
  return;
}

void ReLU::_derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) {
  using namespace autograd;

  this->prev->derive(std::make_shared<ReLU>(seed), out_map);
}

void ReLU::addSubgraph(Agraph_t *graph, Agnode_t *prev) const {
  Agnode_t *curr = agnode(graph, (char *) (std::string("ReLU") + std::to_string(autograd::id_counter++)).c_str(), 1);
  agset(curr, (char *) "label", "ReLU");
  agedge(graph, curr, prev, nullptr, 1);

  this->prev->addSubgraph(graph, curr);
}
