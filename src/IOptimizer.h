#pragma once

#include <CL/cl.h>
class IOptimizer {
  public:
    virtual void optimize(cl_mem layer_parameters, cl_mem layer_gradients) = 0;
};
