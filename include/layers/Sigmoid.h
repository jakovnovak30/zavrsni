/**
 * @file
 * @brief definicija funkcije Sigmoid
 * @author Jakov Novak
 */

#include "../autograd_core/expression.hpp"
#include "../autograd_core/Matrix.h"
#include <memory>

class Sigmoid : public autograd::UnaryOperator<Matrix>, public std::enable_shared_from_this<autograd::Expression<Matrix>> {
  private:
    cl_program program;
    cl_kernel forward_kernel;
  public:
    Sigmoid(std::shared_ptr<Expression<Matrix>> prev);
    ~Sigmoid();

    virtual void eval() override final;
    virtual void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override final;
    virtual void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override final;
};
