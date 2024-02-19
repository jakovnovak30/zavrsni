#include "../Network.h"
#include <CL/cl.h>

class Linear : public Network::ILayer {
  private:
    std::shared_ptr<Matrix> parameters;
    std::shared_ptr<Matrix> biases;
    const size_t in_features, out_features;

    std::shared_ptr<Matrix> last_input;
    std::shared_ptr<Matrix> weight_grad, bias_grad;
    
    cl_program program;
    cl_kernel forward_kernel, bias_kernel;
    cl_kernel bias_grad_kernel, weight_grad_kernel, input_grad_kernel;
    
  public:
    Linear(cl_context context, size_t in_features, size_t out_features);
    Linear(cl_context context, size_t in_features, size_t out_features, bool bias);
    ~Linear();
    
    std::shared_ptr<Matrix> forward(Network &network, std::shared_ptr<Matrix> input_matrix) override final;
    std::shared_ptr<Matrix> backward(Network &network, std::shared_ptr<Matrix> output_grad, std::weak_ptr<IOptimizer> optim) override final;
};
