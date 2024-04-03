#include <gtest/gtest.h>

#include "autograd_core/autograd_util.hpp"

TEST(ScalarTest, TestEvalInt) {
  auto x = autograd::createVariable<int>(10, "x");
  auto expr = x * x + x;

  ASSERT_EQ(expr->getValue(), 110) << "error evaluating expression: x*x + x for x = 10";
};

TEST(ScalarTest, TestEvalFloat) {
  auto x = autograd::createVariable(2.5f, "x");
  auto expr = x / x * x + x;

  ASSERT_EQ(expr->getValue(), 5.f) << "error evaluating expression: x / x * x + x for x = 2.5f";
}

TEST(ScalarTest, TestTwoVars) {
  auto x = autograd::createVariable(3.3f, "x");
  auto y = autograd::createVariable(0.f, "y");
  auto expr = std::static_pointer_cast<autograd::Expression<float>>(std::make_shared<autograd::Exp<float>>(y)) + x;

  ASSERT_EQ(expr->getValue(), 4.3f) << "error evaluating expression: exp(y) + x for (x, y) = (3.3f, 0.f)";
}

TEST(ScalarTest, TestUnivariateDerivative) {
  auto x = autograd::createVariable(5, "x");
  auto constant_2 = autograd::createVariable(2, "const 2", false);
  auto expr = x*x + constant_2*x + autograd::createVariable(1, "const 1", false);

  ASSERT_EQ(expr->grad()["x"]->getValue(), 12) << "error evaluating derivative of: x^2 + 2*x + 1 for x = 5";
}
