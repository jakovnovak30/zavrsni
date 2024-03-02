#include "../src/autograd_core/visualize.hpp"
#include "../src/autograd_core/basic_operations.hpp"
#include <memory>

int main() {
  using namespace autograd;

  std::shared_ptr<Expression<float>> x = std::make_shared<Variable<float>>(10, "x");
  std::shared_ptr<Expression<float>> y = std::make_shared<Variable<float>>(5, "y");
  std::shared_ptr<Expression<float>> z = std::make_shared<Variable<float>>(1, "z");

  auto expr = y + x + x * y * z;

  auto po_x_pa_y = expr->grad()["x"]->grad()["y"];
  visualize(*po_x_pa_y, "out", true);

  auto expr2 = std::dynamic_pointer_cast<Expression<float>>(std::make_shared<Exp<float>>(x)) * x;
}
