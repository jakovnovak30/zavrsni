/**
 * @file
 * @brief definicija modula Linear
 * @author Jakov Novak
 */

#include "../Module.h"
#include <CL/cl.h>

class Linear : public Module {
  private:
    std::shared_ptr<autograd::Variable<Matrix>> parameters;
    std::shared_ptr<autograd::Variable<Matrix>> biases;
    const size_t in_features, out_features;
    
  public:
    Linear(size_t in_features, size_t out_features, bool bias = false);
    
    virtual std::shared_ptr<autograd::Expression<Matrix>> forward(std::shared_ptr<autograd::Expression<Matrix>> ulaz) override final;
};
