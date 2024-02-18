#pragma once

#include "Matrix.h"
class IOptimizer;
#include "Network.h"
#include <CL/cl.h>

class IOptimizer {
  protected:
    static cl_context getContext(Network &network);
    static cl_command_queue getQueue(Network &network);
    static cl_device_id getDevice(Network &network);
  public:
    // funkcija je void zato jer direktno mijenja memoriju na kojoj "žive" parametri
    virtual void optimize(Network &network, Matrix layer_parameters, Matrix layer_gradients) = 0;
};
