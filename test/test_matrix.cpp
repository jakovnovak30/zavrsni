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

  ASSERT_EQ(mat1 + mat2, expected_sum) << "error with matrix summation";
}

TEST(MatrixTest, TestSubMatrix) {
  Matrix mat1 = Matrix({{1, 2}, {3, 4}});
  Matrix mat2 = Matrix({{5, 4}, {7, 8}});
  Matrix expected_sub = Matrix({{4, 2}, {4, 4}});

  ASSERT_EQ(mat2 - mat1, expected_sub) << "error with matrix subtraction";
}

TEST(MatrixTest, TestMulMatrix) {
  Matrix mat1 = Matrix({{1, 2}, {3, 4}});
  Matrix mat2 = Matrix({{5, 4}, {7, 8}});
  Matrix expected_mul = Matrix({{5, 8}, {21, 32}});

  ASSERT_EQ(mat2 * mat1, expected_mul) << "error with elementwise matrix multiplication";
}

TEST(MatrixTest, TestDivMatrix) {
  Matrix mat1 = Matrix({{2, 3}, {9, 8}});
  Matrix mat2 = Matrix({{2, 3}, {3, 4}});
  Matrix expected_div = Matrix({{1, 1}, {3, 2}});

  ASSERT_EQ(mat1 / mat2, expected_div) << "error with elementwise matrix division";
}

TEST(MatrixTest, TestAddScalar) {
  Matrix mat = Matrix({{1, 2}, {2, 1}, {4, 8}});
  const float x = 5;
  Matrix expected = Matrix({{6, 7}, {7, 6}, {9, 13}});

  ASSERT_EQ(mat + x, expected) << "error with matrix-scalar addition";
}

TEST(MatrixTest, TestSubScalar) {
  Matrix mat = Matrix({{1, 2}, {2, 1}, {4, 8}});
  const float x = 5;
  Matrix expected = Matrix({{-4, -3}, {-3, -4}, {-1, 3}});

  ASSERT_EQ(mat - x, expected) << "error with matrix-scalar subtraction";
}

TEST(MatrixTest, TestMulScalar) {
  Matrix mat = Matrix({{5, 9, 9}, {8, 6, 5}});
  const float x = 2;
  Matrix expected = Matrix({{10, 18, 18}, {16, 12, 10}});

  ASSERT_EQ(mat * x, expected) << "error with matrix-scalar multiplication";
}

TEST(MatrixTest, TestDivScalar) {
  Matrix mat = Matrix({{10, 18, 18}, {16, 12, 10}});
  const float x = 2;
  Matrix expected = Matrix({{5, 9, 9}, {8, 6, 5}});

  ASSERT_EQ(mat / x, expected) << "error with matrix-scalar division";
}
