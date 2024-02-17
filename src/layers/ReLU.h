#include "../Network.h"
#include <CL/cl.h>

class ReLU : public Network::ILayer {
  private:
    cl_program program;
    cl_kernel forward_kernel, backward_kernel;
    Matrix last_output;

  public:
    ReLU();
    ~ReLU();
    
    Matrix forward(Network &network, Matrix &input_matrix) override final;
    Matrix backward(Network &network, Matrix &output_grad) override final;
};
