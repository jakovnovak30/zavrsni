#pragma once

#include <CL/cl.h>
#include <array>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <utility>

class Tensor {
private:
  size_t N, M;
  struct Matrix {
    cl_mem data;

    Matrix(cl_mem data);
    ~Matrix();
  };
  std::shared_ptr<Matrix> data_matrix;

public:
  Tensor(cl_mem data, size_t N, size_t M);
  Tensor(std::initializer_list<std::initializer_list<float>> mat);

  Tensor operator+(const Tensor &other) const;
  Tensor operator-(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;
  Tensor operator/(const Tensor &other) const;

  std::shared_ptr<Matrix> getData();

  // samo za matrice manje od 100x100!
  std::string toString();
};
