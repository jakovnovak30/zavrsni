#include "Module.h"
#include "IOptimizer.h"
#include "autograd_core/Matrix.h"
#include <CL/cl.h>
#include <memory>

#ifdef DEBUG
#include <iostream>
#endif

void Module::backward(std::shared_ptr<autograd::Expression<Matrix>> expr, std::weak_ptr<IOptimizer> optim) {
  auto gradients = expr->grad();

  for(auto pair : gradients) {
    #ifdef DEBUG
    std::cout << "[DEBUG] optimizing variable: " << pair.first << std::endl;
    #endif

    Matrix parametri = (*expr)[pair.first]->value;
    optim.lock()->optimize(parametri, pair.second->getValue());
  }

  return;
}
