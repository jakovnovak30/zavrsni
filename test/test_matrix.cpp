#include <gtest/gtest.h>

#include "autograd_core/Matrix.h"

TEST(MatrixTest, TestInvalidInitList) {
  ASSERT_THROW(new Matrix({{3, 1, 2}, {1, -1}}), std::invalid_argument);
}
