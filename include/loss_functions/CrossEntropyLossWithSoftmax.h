#include "autograd_core/expression.hpp"
#include "autograd_core/Matrix.h"
#include <CL/cl.h>

// TODO: dokumentacija
class CrossEntropyLossWithSoftmax : public autograd::BinaryOperator<Matrix> {
  private:
    cl_program program;
    cl_kernel loss_kernel;
  public:
    CrossEntropyLossWithSoftmax(std::shared_ptr<Expression<Matrix>> left, std::shared_ptr<Expression<Matrix>> right);
    ~CrossEntropyLossWithSoftmax();

    virtual void eval() override final;
    virtual void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override final;
    virtual void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override final;
};
