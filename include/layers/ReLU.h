/**
 * @file
 * @brief definicija funkcije ReLU
 * @author Jakov Novak
 */

#include "autograd_core/expression.hpp"
#include "autograd_core/Matrix.h"
#include <CL/cl.h>
#include <memory>

/**
 * @class ReLU
 * @brief klasa koja implementira funkciju "leaky ReLU"
 *
 * Definicija implementirane funkcije je:
 * @f$
 *  f(x) =
 *  \begin{cases}
 *    x, & \text{ako}\ x > 0\\
 *    0.01x, & \text{inače}
 *  \end{cases}
 * @f$
 *
 */
class ReLU : public autograd::UnaryOperator<Matrix>, public std::enable_shared_from_this<autograd::Expression<Matrix>> {
  private:
    /**
     * @brief OpenCL program koji služi za evaluiranje funkcije
     */
    cl_program program;
    /**
     * @brief OpenCL jezgra koja služi za evaluiranje funkcije
     */
    cl_kernel forward_kernel;

  public:
    /**
     * @brief konstruktor prima neki autograd::Expression<Matrix> kao ulaz funkcije
     *
     * @param prev ulaz funkcije
     */
    ReLU(std::shared_ptr<Expression<Matrix>> prev);
    /**
     * @brief destruktor koji oslobađa OpenCL resurse
     */
    ~ReLU();
    
    virtual void eval() override final;
    virtual void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override final;
    virtual void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override final;
};
