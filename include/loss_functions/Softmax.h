#include "autograd_core/expression.hpp"
#include "autograd_core/Matrix.h"
#include <CL/cl.h>

// TODO: dokumentacija
class Softmax : public autograd::UnaryOperator<Matrix> {
  private:
    cl_program program;
    cl_kernel eval_kernel;
  public:
    Softmax(std::shared_ptr<Expression<Matrix>> prev);
    ~Softmax();

    virtual void eval() override final;
    virtual void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override final;
};
