#include "../autograd_core/expression.hpp"
#include "../autograd_core/Matrix.h"
#include <CL/cl.h>

// TODO: prilagodi implementaciju!
class CrossEntropyLoss : public autograd::Expression<Matrix> {
  private:
    cl_program program;
    cl_kernel loss_kernel, grad_kernel;
  public:
    CrossEntropyLoss();
    ~CrossEntropyLoss() = default;

    virtual void eval() override final;
    virtual void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override final;
};
