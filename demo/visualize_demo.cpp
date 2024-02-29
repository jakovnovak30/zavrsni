#include "../src/autograd_core/visualize.hpp"

int main() {
  autograd::Variable<float> x = autograd::Variable<float>(10, "x");
  autograd::Variable<float> y = autograd::Variable<float>(5, "y");

  auto expr = x * y + x;

  autograd::visualize(expr, "out");
}
