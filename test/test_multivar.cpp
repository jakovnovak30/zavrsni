#include <gtest/gtest.h>
#include <memory>

#include "autograd_core/autograd_util.hpp"

TEST(TestMultivar, TestEvalAddMul) {
  auto x = autograd::createVariable(10, "x");
  auto y = autograd::createVariable(5, "y");
  auto expr = x * y + y;

  ASSERT_EQ(expr->getValue(), 55) << "error evaluating expression: x * y + y for (x, y) = (10, 5)";
}

TEST(TestMultivar, TestGradAddMul) {
  auto x = autograd::createVariable(10, "x");
  auto y = autograd::createVariable(5, "y");
  auto expr = x * y + y;

  EXPECT_EQ(expr->grad()["x"]->getValue(), 5) << "error deriving expression: x * y + y (for x)";
  EXPECT_EQ(expr->grad()["y"]->getValue(), 11) << "error deriving expression: x * y + y (for y)";
}

TEST(TestMultivar, TestEvalSubAdd) {
  auto x = autograd::createVariable(1, "x");
  auto y = autograd::createVariable(2, "y");
  auto z = autograd::createVariable(3, "z");

  auto expr = x + y - z;
  
  ASSERT_EQ(expr->getValue(), 0) << "error evaluating expression: x + y - z for (1, 2, 3)";
}

TEST(TestMultivar, TestGradSubAdd) {
  auto x = autograd::createVariable(1, "x");
  auto y = autograd::createVariable(2, "y");
  auto z = autograd::createVariable(3, "z");

  auto expr = x + y - z;
  
  EXPECT_EQ(expr->grad()["x"]->getValue(), 1) << "error deriving expression: x + y - z (for x)";
  EXPECT_EQ(expr->grad()["y"]->getValue(), 1) << "error deriving expression: x + y - z (for y)";
  EXPECT_EQ(expr->grad()["z"]->getValue(), -1) << "error deriving expression: x + y - z (for z)";
}

TEST(TestMultivar, TestEvalSubMul) {
  auto x = autograd::createVariable(4, "x");
  auto y = autograd::createVariable(3, "y");
  auto z = autograd::createVariable(5, "z");

  auto expr = x * y + z * x;

  ASSERT_EQ(expr->getValue(), 32) << "error evaluating expression: x * y + z * x for (x, y, z) = (4, 3, 5)";
}

TEST(TestMultivar, TestGradSubMul) {
  auto x = autograd::createVariable(4, "x");
  auto y = autograd::createVariable(3, "y");
  auto z = autograd::createVariable(5, "z");

  auto expr = x * y + z * x;

  EXPECT_EQ(expr->grad()["x"]->getValue(), 8) << "error deriving expression: x * y+ z * x (for x)";
  EXPECT_EQ(expr->grad()["y"]->getValue(), 4) << "error deriving expression: x * y+ z * x (for y)";
  EXPECT_EQ(expr->grad()["z"]->getValue(), 4) << "error deriving expression: x * y+ z * x (for z)";
}

TEST(TestMultivar, TestEvalExpMul) {
  auto x = autograd::createVariable(1.f, "x");
  auto y = autograd::createVariable(2.f, "y");
  auto expr = std::static_pointer_cast<autograd::Expression<float>>(std::make_shared<autograd::Exp<float>>(x)) * y;

  ASSERT_FLOAT_EQ(expr->getValue(), std::exp(1) * 2);
}

TEST(TestMultivar, TestGradExpMul) {
  auto x = autograd::createVariable(1.f, "x");
  auto y = autograd::createVariable(2.f, "y");
  auto expr = std::static_pointer_cast<autograd::Expression<float>>(std::make_shared<autograd::Exp<float>>(x)) * y;

  EXPECT_FLOAT_EQ(expr->grad()["x"]->getValue(), std::exp(1) * 2);
  EXPECT_FLOAT_EQ(expr->grad()["x"]->grad()["x"]->getValue(), std::exp(1) * 2);

  EXPECT_FLOAT_EQ(expr->grad()["y"]->getValue(), std::exp(1));
}
