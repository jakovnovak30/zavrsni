#include <gtest/gtest.h>

#include "autograd_core/autograd_util.hpp"

TEST(TestMultivar, TestEvalAddMul) {
  auto x = autograd::createVariable(10, "x");
  auto y = autograd::createVariable(5, "y");
  auto expr = x * y + y;

  ASSERT_EQ(expr->getValue(), 55);
}

TEST(TestMultivar, TestGradAddMul) {
  auto x = autograd::createVariable(10, "x");
  auto y = autograd::createVariable(5, "y");
  auto expr = x * y + y;

  EXPECT_EQ(expr->grad()["x"]->getValue(), 5);
  EXPECT_EQ(expr->grad()["y"]->getValue(), 11);
}
