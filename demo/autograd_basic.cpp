#include "../src/autograd_core/expression.hpp"
#include <iostream>
#include "../src/Util.h"
#include "../src/autograd_core/Tensor.h"

int main() {
  autograd::Variable x = autograd::Variable<float>(1);
  autograd::Variable y = autograd::Variable<float>(2);
  autograd::Variable z = autograd::Variable<float>(5);

  auto expr = autograd::Exp(-((x + y) * (x + y) * x - z*autograd::Variable<float>(2)));
  // x * (x + y)^2
  expr.eval();

  // derivacija po x je: 2*(x + y) + (x + y)^2 = 2*3 + 9 = 15
  // derivacija po y je: x*2*(x + y) = 2*3 = 6
  expr.autograd::Expression<float>::derive();

  std::cout << expr.value << std::endl;
  std::cout << "po x: " << x.partial << std::endl << "po y: " << y.partial << std::endl;
  std::cout << "po z: " << z.partial << std::endl;

  autograd::Variable k = autograd::Variable<float>(2);
  auto test2 = autograd::Exp(k) * k;

  test2.eval();
  test2.autograd::Expression<float>::derive();

  std::cout << "test2: " << test2.value << std::endl;
  std::cout << "derivacija " << k.partial << std::endl;

  initCL_nvidia();
  Tensor ten1 = {
                  { 5.f, 4.f, 3.f },
                  { 3.f, 2.f, 1.f}
                };
  Tensor ten2 = {
                    { 1.f, 2.f, 3.f },
                    { 3.f, 2.f, 1.f }
                  };

  std::cout << (ten1 * ten2).toString() << std::endl;
}
