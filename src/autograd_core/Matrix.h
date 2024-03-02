#pragma once

#include <CL/cl.h>
#include <array>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <utility>

class Matrix {
private:
  cl_mem data;
  size_t N, M;
public:
  Matrix(cl_mem data, size_t N, size_t M);
  Matrix(std::initializer_list<std::initializer_list<float>> mat);
  Matrix(const float x);
  ~Matrix();

  void operator+=(const Matrix &other);
  Matrix operator+(const Matrix &other) const;
  Matrix operator-(const Matrix &other) const;
  Matrix operator*(const Matrix &other) const;
  Matrix operator/(const Matrix &other) const;

  // samo za matrice manje od 100x100!
  std::string toString();
};
