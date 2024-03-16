#include "../src/autograd_core/visualize.hpp"
#include "../src/autograd_core/basic_operations.hpp"
#include "../src/autograd_core/autograd_util.hpp"
#include <memory>
#include <iostream>

int main() {
  using namespace autograd;

  auto x = createVariable(10.f, "x");
  auto y = createVariable(5.f, "y");
  auto z = createVariable(1.f, "z");

  auto expr = y + x + x * y * z;

  auto po_x_pa_y = expr->grad()["x"]->grad()["y"];
  // visualize(*po_x_pa_y, "out", true);

  auto expr2 = mathFunc<float>(Exp(x)) * x + expr;
  visualize(*expr2->grad()["y"], "out", true);
  std::cout << expr2->grad()["y"]->getValue() << std::endl;
  // visualize(*expr2->grad()["x"]->grad()["y"], "out", true);
}
