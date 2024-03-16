#include "../src/autograd_core/expression.hpp"
#include "../src/autograd_core/basic_operations.hpp"
#include "../src/autograd_core/visualize.hpp"
#include "../src/autograd_core/autograd_util.hpp"
#include <iostream>
#include <memory>

int main() {
  using namespace autograd;
  auto x = createVariable(1.f, "x");
  auto y = createVariable(2.f, "y");
  auto z = createVariable(5.f, "z");

  auto expr = std::dynamic_pointer_cast<Expression<float>>(
    std::make_shared<Exp<float>>(-(std::dynamic_pointer_cast<Expression<float>>(std::make_shared<Exp<float>>(x + y)) * x - z))) + z;

  auto expr_gradient = expr->grad();
  std::cout << expr->getValue() << std::endl;
  std::cout << "po x: " << expr_gradient["x"]->getValue() << std::endl;
  std::cout << "po y: " << expr_gradient["y"]->getValue() << std::endl;
  std::cout << "po z: " << expr_gradient["z"]->getValue() << std::endl;

  std::shared_ptr<Expression<float>> k = std::make_shared<Variable<float>>(2, "k");
  auto test2 = std::dynamic_pointer_cast<Expression<float>>(std::make_shared<Exp<float>>(k)) * k;

  std::cout << "test2: " << test2->getValue() << std::endl;
  std::cout << "derivacija " << test2->grad()["k"]->getValue() << std::endl;
}
