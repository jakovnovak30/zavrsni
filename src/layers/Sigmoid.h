#include "../Network.h"

class Sigmoid : public Network::ILayer {
  private:
    cl_program program;
    cl_kernel forward_kernel, backward_kernel;
    Matrix last_output;

  public:
    Sigmoid();
    ~Sigmoid();

    Matrix forward(Network &network, Matrix &input_matrix) override final;
    Matrix backward(Network &network, Matrix &output_grad, IOptimizer *optim) override final;
};
