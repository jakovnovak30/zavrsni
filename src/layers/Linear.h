#include "../Network.h"
#include <CL/cl.h>

// TODO: prilagodi impl. datoteku !!!
// P.S. koja je pravilna nadklasa za ovo??
class Linear : public autograd::Expression<Matrix> {
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
    Linear(size_t in_features, size_t out_features);
    Linear(size_t in_features, size_t out_features, bool bias);
    ~Linear();
    
    virtual void eval() override final;
    virtual void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override final;
};
