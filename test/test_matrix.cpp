#include <gtest/gtest.h>

#include "Util.h"
#include "autograd_core/Matrix.h"

TEST(MatrixTest, TestInvalidInitList) {
  initCL_nvidia();
  ASSERT_THROW(new Matrix({{3, 1, 2}, {1, -1}}), std::invalid_argument);
}

TEST(MatrixTest, TestAddMatrix) {
  Matrix mat1 = Matrix({{1, 2}, {3, 4}});
  Matrix mat2 = Matrix({{5, 4}, {7, 8}});
  Matrix expected_sum = Matrix({{6, 6}, {10, 12}});

  ASSERT_EQ((mat1 + mat2).toString(), expected_sum.toString()) << "error with matrix summation";
}

TEST(MatrixTest, TestSubMatrix) {
  Matrix mat1 = Matrix({{1, 2}, {3, 4}});
  Matrix mat2 = Matrix({{5, 4}, {7, 8}});
  Matrix expected_sub = Matrix({{4, 2}, {4, 4}});

  ASSERT_EQ((mat2 - mat1).toString(), expected_sub.toString());
}

TEST(MatrixTest, TestMulMatrix) {
  Matrix mat1 = Matrix({{1, 2}, {3, 4}});
  Matrix mat2 = Matrix({{5, 4}, {7, 8}});
  Matrix expected_mul = Matrix({{5, 8}, {21, 32}});

  ASSERT_EQ((mat2 * mat1).toString(), expected_mul.toString());
}

TEST(MatrixTest, TestDivMatrix) {
  Matrix mat1 = Matrix({{2, 3}, {9, 8}});
  Matrix mat2 = Matrix({{2, 3}, {3, 4}});
  Matrix expected_div = Matrix({{1, 1}, {3, 2}});

  ASSERT_EQ((mat1 / mat2).toString(), expected_div.toString());
}
