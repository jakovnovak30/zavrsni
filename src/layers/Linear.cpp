#include "layers/Linear.h"
#include "autograd_core/matrix_operations.hpp"
#include "Util.h"

#include <CL/cl.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>

#ifdef DEBUG
#include <iostream>
#endif

inline float gen_uniform(float k) {
  static bool called = false;
  if(!called) {
    srand((unsigned) time(nullptr));
    called = true;
  }

  float HI = sqrt(k);
  float LO = -HI;
  return LO + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/(HI-LO)));
}

static size_t bias_counter = 0;
static size_t weight_counter = 0;

// initialize values from U(-sqrt(k), +sqrt(k)), where k = 1 / in_features
Linear::Linear(const size_t in_features, const size_t out_features, bool bias) : in_features{ in_features }, out_features{ out_features }
{
  int _err;
  float *host_ptr = new float[in_features*out_features];
  for(size_t i=0;i < in_features*out_features;i++) {
    host_ptr[i] = gen_uniform(1.0f / in_features);
  }
  cl_mem parameters_cl = clCreateBuffer(globalContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, in_features * out_features * sizeof(float), host_ptr, &_err);
  checkError(_err);
  delete[] host_ptr;
  this->parameters = std::make_shared<autograd::Variable<Matrix>>(Matrix(parameters_cl, out_features, in_features), "weights" + std::to_string(weight_counter++), true);

  if(bias) {
    float *host_ptr = new float[out_features];
    for(size_t i=0;i < out_features;i++)
      host_ptr[i] = gen_uniform(1.0f / in_features);
    cl_mem biases_cl = clCreateBuffer(globalContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, out_features * sizeof(float), host_ptr, &_err);
    checkError(_err);
    delete[] host_ptr;

    this->biases = std::make_shared<autograd::Variable<Matrix>>(Matrix(biases_cl, 1, out_features), "bias" + std::to_string(bias_counter++), true);
  }
  else {
    this->biases = nullptr;
  }
}

std::shared_ptr<autograd::Expression<Matrix>> Linear::forward(std::shared_ptr<autograd::Expression<Matrix>> ulaz) {
  std::shared_ptr<autograd::Expression<Matrix>> expr = std::make_shared<autograd::MatrixMultply>(ulaz, this->parameters);

  if(this->biases != nullptr) {
    expr = std::make_shared<autograd::MatrixVectorAdd>(expr, this->biases);
  }

  return expr;
}
