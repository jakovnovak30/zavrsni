#include "../Network.h"
#include <CL/cl.h>

class Linear : public Network::ILayer {
  private:
    cl_mem parameters;
    cl_mem biases;
    const size_t in_features, out_features;
    
    cl_program program;
    cl_kernel forward_kernel, backward_kernel, bias_kernel;
    
  public:
    Linear(cl_context context, size_t in_features, size_t out_features);
    Linear(cl_context context, size_t in_features, size_t out_features, bool bias);
    ~Linear();
    
    Matrix forward(Network &network, Matrix &input_matrix) override final;
    Matrix backward(Network &network, Matrix &output_grad) override final;
};
