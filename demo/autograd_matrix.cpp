/**
 * @file
 * @brief demonstracijski program u kojem se prikazuju osnovne funkcionalnosti autograda zajedno s matricama (koje su implementirane pomoću OpenCL-a)
 */

#include "Util.h"
#include "layers/Sigmoid.h"
#include "layers/ReLU.h"
#include "autograd_core/Matrix.h"
#include "autograd_core/autograd_util.hpp"
#include "autograd_core/visualize.hpp"
#include "autograd_core/matrix_operations.hpp"
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

  
  auto mat_expr = std::make_shared<Sigmoid>(mat_x);


  // std::cout << mat_expr->grad()["y"]->getValue().toString() << std::endl;
  visualize(*mat_expr->grad()["x"], "/tmp/vec_test.png", true);
  // visualize(*mat_expr, "/tmp/vec_test.png", true);

  // test
  std::cout << "matrica x * 2 = " << (mat_x->getValue() * Matrix(2)).toString() << std::endl;

  freeCL();
}
