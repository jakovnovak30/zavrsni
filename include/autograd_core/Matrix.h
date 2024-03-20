#pragma once

#include <CL/cl.h>
#include <initializer_list>
#include <memory>

#include "Util.h"

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
  // utility function...
  inline cl_kernel loadKernel(const Matrix &other, const std::string &name) const;

public:
  Matrix(cl_mem data, size_t N, size_t M);
  Matrix(std::initializer_list<std::initializer_list<float>> mat);
  Matrix(const float x); // za skalare!
  Matrix();

  void operator+=(const Matrix &other);
  Matrix operator+(const Matrix &other) const;
  Matrix operator-(const Matrix &other) const;
  Matrix operator-() const;
  Matrix operator*(const Matrix &other) const;
  Matrix operator/(const Matrix &other) const;

  // samo za matrice manje od 100x100!
  std::string toString() const;

  // geteri
  size_t getN();
  size_t getM();
  std::shared_ptr<opencl_data> data;
};
