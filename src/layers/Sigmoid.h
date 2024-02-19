#include "../Network.h"

class Sigmoid : public Network::ILayer {
  private:
    cl_program program;
    cl_kernel forward_kernel, backward_kernel;
    std::shared_ptr<Matrix> last_output;

  public:
    Sigmoid();
    ~Sigmoid();

    std::shared_ptr<Matrix> forward(Network &network, std::shared_ptr<Matrix> input_matrix) override final;
    std::shared_ptr<Matrix> backward(Network &network, std::shared_ptr<Matrix> output_grad, std::weak_ptr<IOptimizer> optim) override final;
};
