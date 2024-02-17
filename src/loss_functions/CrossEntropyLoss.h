#include "../ILossFunc.h"
#include <CL/cl.h>

class CrossEntropyLoss : public ILossFunc {
  private:
    cl_program program;
    cl_kernel loss_kernel, grad_kernel;
  public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();

    Matrix calculate_loss(Network &network, Matrix &input, Matrix &expected) override final;
    Matrix calculate_gradient(Network &network, Matrix &input, Matrix &expected) override final;
};
