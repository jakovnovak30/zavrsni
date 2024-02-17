#pragma once

#include <list>
#include <CL/cl.h>

class ILossFunc {
public:
  virtual float calculate_loss(std::list<float> expected, std::list<float> actual);
  // TODO: nekaj s derivacijom
};

class IOptimizer {
  public:
    virtual void optimize(cl_mem layer_parameters, cl_mem layer_gradients);
};

class Network {
  public:
    class ILayer {
      public:
        // calculate the result of forward propagation for the current layer
        // returns a handle to the memory used for the result, waits internally for the calculation
        virtual cl_mem forward(cl_mem input_buffer);
        // TODO
        virtual void backward(ILossFunc loss_func, IOptimizer optim);
    };

    Network(cl_context context, cl_device_id device, std::list<ILayer> layers);
    ~Network();
    
    cl_mem forward(cl_mem input_buffer);
    cl_mem forward(void *input_buffer, size_t N);
    void backward(ILossFunc loss_func, IOptimizer optim);

  private:
    cl_command_queue queue;
    cl_context context;
    std::list<ILayer> layers;

};

void checkError(int value);
