#include "../src/autograd_core/visualize.hpp"
#include <iostream>

int main() {
  autograd::Variable<float> x = autograd::Variable<float>(10, "x");
  autograd::Variable<float> y = autograd::Variable<float>(5, "y");
  autograd::Variable<float> z = autograd::Variable<float>(0, "z");

  auto expr = x * y + x - autograd::Exp(-(x * z));

  autograd::visualize(expr, "out", true);

  expr.eval();
  std::cout << "evaluation: " << expr.value << std::endl;
  std::cout << "delta(expr) / delta(x): " << x.partial << std::endl;
}
