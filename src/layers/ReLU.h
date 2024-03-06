#include "../Network.h"
#include <CL/cl.h>

// TODO: implementacija
class ReLU : public autograd::UnaryOperator<Matrix> {
  private:
    cl_program program;
    cl_kernel forward_kernel, backward_kernel;
    std::shared_ptr<Matrix> last_output;

  public:
    ReLU();
    ~ReLU();
    
    virtual void eval() override final;
    virtual void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override final;
};
