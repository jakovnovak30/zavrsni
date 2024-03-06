#include "../Network.h"

// TODO: implementacija
class Sigmoid : public autograd::UnaryOperator<Matrix> {
  private:
    cl_program program;
    cl_kernel forward_kernel, backward_kernel;
  public:
    Sigmoid();
    ~Sigmoid();

    virtual void eval() override final;
    virtual void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override final;
};
