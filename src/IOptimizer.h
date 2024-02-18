#pragma once

#include "Matrix.h"
#include <CL/cl.h>

class IOptimizer {
  public:
    virtual void optimize(Matrix layer_parameters, Matrix layer_gradients) = 0;
};
