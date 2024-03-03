#include "Matrix.h"
#include "../Util.h"
#include <CL/cl.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

static cl_program basicOpsProgram = nullptr;
static cl_kernel addKernel = nullptr;
static cl_kernel subKernel = nullptr;
static cl_kernel mulKernel = nullptr;
static const char *srcCode[] =
  {
      #include "../kernels/BasicMatrix.cl"
  };
static const size_t srcLen[] = { strlen(srcCode[0]) };

Matrix::Matrix(cl_mem data, size_t N, size_t M) : data(std::make_shared<opencl_data>(data)), N(N), M(M) { }

// TODO: dodaj podrsku za skalarne operacije!
Matrix::Matrix(const float x) : Matrix({{ x, x }, { x, x }}) { }

Matrix::Matrix() : data(nullptr) {
  this->N = 0;
  this->M = 0;
}

Matrix::Matrix(std::initializer_list<std::initializer_list<float>> mat) : data(std::make_shared<opencl_data>(mat.size(), mat.begin()->size())) {
  // find height / width
  const size_t height = mat.size();
  const size_t width = mat.begin()->size();
  for(auto it = mat.begin();it != mat.end();it++) {
    if(width != it->size()) {
      throw std::invalid_argument("matrix is invalid");
    }
  }
  
  float *buffer = new float[width * height];
  size_t counter = 0;
  for(auto it1 = mat.begin();it1 != mat.end();it1++) {
    for(auto it2 = it1->begin();it2 != it1->end();it2++) {
      buffer[counter++] = *it2;
    }
  }

  checkError(clEnqueueWriteBuffer(globalQueue, *this->data, CL_FALSE, 0, height * width * sizeof(float), buffer, 0, nullptr, nullptr));
  this->N = height;
  this->M = width;
  delete[] buffer;
}

std::string Matrix::toString() {
  if(this->N > 100 || this->M > 100) {
    throw std::invalid_argument("matrix is too big to display!");
  }
  std::stringstream ss;

  int _err;
  cl_event event = clCreateUserEvent(globalContext, &_err);
  checkError(_err);
  float *host_ptr = new float[this->N * this->M];
  checkError(clEnqueueReadBuffer(globalQueue, *this->data, CL_FALSE, 0, this->N * this->M * sizeof(float), host_ptr, 0, nullptr, &event));
  clWaitForEvents(1, &event);
  clReleaseEvent(event);

  ss << "Matrix(";
  for(size_t i = 0;i < this->N;i++) {
    ss << "{";
    for(size_t j = 0;j < this->M;j++) {
      ss << " " << host_ptr[i * this->M + j];
    }
    if(i == this->N - 1)
      ss << " }})" << std::endl;
    else
      ss << " }" << std::endl;
  }

  delete[] host_ptr;
  return ss.str();
}

Matrix Matrix::operator+(const Matrix &other) const {
  int _err;
  cl_mem out_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, this->N * this->M * sizeof(float), nullptr, &_err);
  checkError(_err);

  buildIfNeeded(&basicOpsProgram, &addKernel, "matrixAdd", srcCode, srcLen);

  checkError(clSetKernelArg(addKernel, 0, sizeof(float *), &this->data->data));
  checkError(clSetKernelArg(addKernel, 1, sizeof(float *), &other.data->data));
  checkError(clSetKernelArg(addKernel, 2, sizeof(float *), &out_buffer));
  checkError(clSetKernelArg(addKernel, 3, sizeof(const int), &this->M));

  const size_t global_work_size[] = { this->N, this->M };
  cl_event test_event = clCreateUserEvent(globalContext, &_err);
  checkError(_err);
  checkError(clEnqueueNDRangeKernel(globalQueue, addKernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, &test_event));
  checkError(clWaitForEvents(1, &test_event));

  return Matrix(out_buffer, this->N, this->M);
}

Matrix Matrix::operator-(const Matrix &other) const {
  int _err;
  cl_mem out_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, this->N * this->M * sizeof(float), nullptr, &_err);
  checkError(_err);

  buildIfNeeded(&basicOpsProgram, &subKernel, "matrixSub", srcCode, srcLen);

  clSetKernelArg(addKernel, 0, sizeof(float *), &this->data->data);
  clSetKernelArg(addKernel, 1, sizeof(float *), &other.data->data);
  clSetKernelArg(addKernel, 2, sizeof(float *), &out_buffer);
  clSetKernelArg(addKernel, 3, sizeof(const int), &this->M);

  const size_t global_work_size[] = { this->N, this->M };
  clEnqueueNDRangeKernel(globalQueue, subKernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);

  return Matrix(out_buffer, this->N, this->M);
}

Matrix Matrix::operator*(const Matrix &other) const {
  int _err;
  cl_mem out_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, this->N * this->M * sizeof(float), nullptr, &_err);
  checkError(_err);

  buildIfNeeded(&basicOpsProgram, &mulKernel, "matrixMulScalar", srcCode, srcLen);

  clSetKernelArg(mulKernel, 0, sizeof(float *), &this->data->data);
  clSetKernelArg(mulKernel, 1, sizeof(float *), &other.data->data);
  clSetKernelArg(mulKernel, 2, sizeof(float *), &out_buffer);
  clSetKernelArg(mulKernel, 3, sizeof(const int), &this->M);

  const size_t global_work_size[] = { this->N, this->M };
  clEnqueueNDRangeKernel(globalQueue, mulKernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);

  return Matrix(out_buffer, this->N, this->M);
}

Matrix Matrix::operator/(const Matrix &other) const {
  return {{ 1.f }};
}
