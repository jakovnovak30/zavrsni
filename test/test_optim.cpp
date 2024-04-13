#include<gtest/gtest.h>

#include "optimizers/SGD.h"

TEST(TestOptimizers, TestSGDSingleIteration) {

  Matrix parameters = Matrix({{4.f, 5.f, 7.f}, {1.2f, 2.4f, 3.6f}, {1.2f, 2.4f, 3.6f}, {4.f, 5.f, 7.f}});
  Matrix gradients = Matrix({{.2f, .25f, .1f}, {.1f, .25f, .2f}, {.1f, .25f, .2f}, {.2f, .25f, .1f}});

  Matrix expected = Matrix({{3.6f, 4.5f, 6.8f}, {1.f, 1.9f, 3.2f}, {1.f, 1.9f, 3.2f}, {3.6f, 4.5f, 6.8f}});

  IOptimizer *optim = new SGD(2.f);
  optim->optimize(parameters, gradients);
  delete optim;

  ASSERT_EQ(expected.toString(), parameters.toString());
}
