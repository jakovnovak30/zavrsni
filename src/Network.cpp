#include "Network.h"
#include "IOptimizer.h"
#include "autograd_core/Matrix.h"
#include <CL/cl.h>
#include <memory>

#ifdef DEBUG
#include <iostream>
#endif


void Network::backward(std::shared_ptr<autograd::Expression<Matrix>> expr, std::weak_ptr<IOptimizer> optim) {
  auto gradients = expr->grad();

  for(auto pair : gradients) {
    #ifdef DEBUG
    std::cout << "[DEBUG] optimizing variable: " << pair.first << std::endl;
    #endif

    // TODO: di su parametri??
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wall"
    optim.lock()->optimize(nullptr, pair.second->getValue());
    #pragma GCC diagnostic pop
  }

  return;
}
