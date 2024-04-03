/**
 * @file
 * @brief definicija funkcije Sigmoid
 * @author Jakov Novak
 */

#include "autograd_core/expression.hpp"
#include "autograd_core/Matrix.h"
#include <memory>

/**
 * @class Sigmoid
 * @brief implementacije sigmoid funkcije: \f$ f(x) = \frac{1}{1 - e^{-x}} \f$
 *
 */
class Sigmoid : public autograd::UnaryOperator<Matrix>, public std::enable_shared_from_this<autograd::Expression<Matrix>> {
  private:
    /**
     * @brief OpenCL program koji se koristi
     */
    cl_program program;
    /**
     * @brief OpenCL jezgra koja se koristi
     */
    cl_kernel forward_kernel;
  public:
    /**
     * @brief konstruktor koji prima izraz \f$ x \f$
     *
     * @param prev ulaz funkcije tipa autograd::Expression<Matrix>
     */
    Sigmoid(std::shared_ptr<Expression<Matrix>> prev);
    /**
     * @brief destruktor koji oslobađa OpenCL resurse
     */
    ~Sigmoid();

    virtual void eval() override final;
    virtual void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override final;
    virtual void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override final;
};
