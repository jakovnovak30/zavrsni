#pragma once

#include <CL/cl.h>
#include <array>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <utility>

#include "../Util.h"

class Matrix {
private:
  struct opencl_data {
    opencl_data(cl_mem data) : data{ data } { }
    opencl_data(const int N, const int M) {
      int _err;
      this->data = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, N * M * sizeof(float), nullptr, &_err);
      checkError(_err);
    }
    ~opencl_data() { if(this->data != nullptr) checkError(clReleaseMemObject(this->data)); }
    cl_mem data;

    operator cl_mem () {
      return this->data;
    }
  };
  size_t N, M;

public:
  Matrix(cl_mem data, size_t N, size_t M);
  Matrix(std::initializer_list<std::initializer_list<float>> mat);
  Matrix(const float x); // za skalare!
  Matrix();

  void operator+=(const Matrix &other);
  Matrix operator+(const Matrix &other) const;
  Matrix operator-(const Matrix &other) const;
  Matrix operator*(const Matrix &other) const;
  Matrix operator/(const Matrix &other) const;

  // samo za matrice manje od 100x100!
  std::string toString();

  // geteri
  const size_t getN();
  const size_t getM();
  std::shared_ptr<opencl_data> data;
};
