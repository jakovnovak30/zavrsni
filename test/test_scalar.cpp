#include <gtest/gtest.h>
#include <memory>

#include "autograd_core/autograd_util.hpp"
#include "autograd_core/expression.hpp"

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

TEST(ScalarTest, TestEvalNeg) {
  auto x = autograd::createVariable(10, "x");
  auto expr1 = -x;
  auto expr2 = -std::static_pointer_cast<autograd::Expression<int>>(expr1);

  EXPECT_EQ(expr1->getValue(), -10) << "error evaluating expression: -x for x = 10";
  EXPECT_EQ(expr2->getValue(), 10) << "error evaluating expression: -(-x) for x = 10";
}

TEST(ScalarTest, TestGradNeg) {
  auto x = autograd::createVariable(10, "x");
  auto expr1 = -x;
  auto expr2 = -std::static_pointer_cast<autograd::Expression<int>>(expr1);

  EXPECT_EQ(expr1->grad()["x"]->getValue(), -1) << "error deriving f(x) = -x for x = 10";
  EXPECT_EQ(expr2->grad()["x"]->getValue(), 1) << "error deriving f(x) = -(-x) for x = 10";
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

TEST(ScalarTest, TestUnvariateSecondDerivative) {
  auto x = autograd::createVariable(7, "x");
  auto expr = x * x * x + x * x;

  ASSERT_EQ(expr->grad()["x"]->grad()["x"]->getValue(), 44) << "error evaluating second derivative of: x^3 + x^2 for x = 7";
}

TEST(ScalarTest, TestUnivariateThirdDerivative) {
  auto x = autograd::createVariable(0.f, "x");
  auto expr = std::static_pointer_cast<autograd::Expression<float>>(std::make_shared<autograd::Exp<float>>(x)) - x*x*x;

  ASSERT_EQ(expr->grad()["x"]->grad()["x"]->grad()["x"]->getValue(), -5.f) << "error evaluting third derivative of: exp(x) - x^3 for x = 0.f";
}

TEST(ScalarTest, TestTwoVarsDerivative) {
  auto x = autograd::createVariable(3.3f, "x");
  auto y = autograd::createVariable(0.f, "y");
  auto expr = std::static_pointer_cast<autograd::Expression<float>>(std::make_shared<autograd::Exp<float>>(y)) + x * x;

  EXPECT_EQ(expr->grad()["x"]->getValue(), 6.6f) << "error evaluating partial (x) of expression: exp(y) + x for (x, y) = (3.3f, 0.f)";
  EXPECT_EQ(expr->grad()["y"]->getValue(), 1.f) << "error evaluating partial (y) of expression: exp(y) + x for (x, y) = (3.3f, 0.f)";
}
