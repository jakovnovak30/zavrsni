#include "../Network.h"
#include <CL/cl.h>
#include <memory>

class ReLU : public autograd::UnaryOperator<Matrix>, public std::enable_shared_from_this<autograd::Expression<Matrix>> {
  private:
    cl_program program;
    cl_kernel forward_kernel;

  public:
    ReLU(std::shared_ptr<Expression<Matrix>> prev);
    ~ReLU();
    
    virtual void eval() override final;
    virtual void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override final;
    virtual void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override final;
};
