#include "../ILossFunc.h"
#include <CL/cl.h>

class CrossEntropyLoss : public ILossFunc {
  private:
    cl_program program;
    cl_kernel loss_kernel, grad_kernel;
  public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();

    std::shared_ptr<Matrix> calculate_loss(Network &network, std::shared_ptr<Matrix> input, std::shared_ptr<Matrix> expected) override final;
    std::shared_ptr<Matrix> calculate_gradient(Network &network, std::shared_ptr<Matrix> input, std::shared_ptr<Matrix> expected) override final;
};
