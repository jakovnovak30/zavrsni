/**
 * @file
 * @brief implementacije generičke kompozicije funkcija Module
 * @author Jakov Novak
 */

#include "Module.h"
#include "IOptimizer.h"
#include "autograd_core/Matrix.h"
#include "autograd_core/visualize.hpp"
#include <CL/cl.h>
#include <memory>

void Module::backward(std::shared_ptr<autograd::Expression<Matrix>> expr, std::weak_ptr<IOptimizer> optim) {
  auto gradients = expr->grad();

  for(auto pair : gradients) {
    Matrix parametri = (*expr)[pair.first]->value;
    optim.lock()->optimize(parametri, pair.second->getValue());
  }

  return;
}
