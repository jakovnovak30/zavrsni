#include "../src/Util.h"
#include "../src/layers/Sigmoid.h"
#include "../src/layers/ReLU.h"
#include "../src/autograd_core/Matrix.h"
#include "../src/autograd_core/autograd_util.hpp"
#include "../src/autograd_core/basic_operations.hpp"
#include "../src/autograd_core/visualize.hpp"
#include "../src/autograd_core/matrix_operations.hpp"
#include <iostream>

int main() {
  using namespace autograd;

  initCL_nvidia();
  Matrix mat1 = {
                  { 5.f, 4.f, 3.f },
                  { 3.f, 2.f, 1.f}
                };
  Matrix mat2 = {
                    { 2.f, 2.f, 3.f },
                    { 3.f, 2.f, 1.f }
                  };

  // std::cout << (mat1 + mat2).toString() << std::endl;

  auto mat_x = createVariable(Matrix{{1.f, 2.f}, {1.f, 5.f}}, "x");
  auto mat_y = createVariable(Matrix{{2.f, 3.f}, {3.f, 2.f}}, "y");

  
  auto mat_expr = std::make_shared<ReLU>(mat_x);


  // std::cout << mat_expr->grad()["y"]->getValue().toString() << std::endl;
  visualize(*mat_expr->grad()["x"], "/tmp/vec_test.png", true);
  // visualize(*mat_expr, "/tmp/vec_test.png", true);

  // test
  std::cout << "matrica x * 2 = " << (mat_x->getValue() * Matrix(2)).toString() << std::endl;

  freeCL();
}
