#include "Matrix.h"
#include "../Util.h"
#include <CL/cl.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

static cl_program basicOpsProgram = nullptr;
static std::unordered_map<std::string, cl_kernel> basicKernels = std::unordered_map<std::string, cl_kernel>();
static const char *srcCode[] =
  {
      #include "../kernels/BasicMatrix.cl"
  };
static const size_t srcLen[] = { strlen(srcCode[0]) };

static cl_program scalarOpsProgram = nullptr;
static std::unordered_map<std::string, cl_kernel> scalarKernels = std::unordered_map<std::string, cl_kernel>();
static const char *scalarSrcCode[] =
  {
      #include "../kernels/BasicScalarMatrix.cl"
  };
static const size_t scalarSrcLen[] = { strlen(scalarSrcCode[0]) };

Matrix::Matrix(cl_mem data, size_t N, size_t M) : data(std::make_shared<opencl_data>(data)), N(N), M(M) { }

// skalar u kontekstu linearne algebre -> matrica 1x1 radi jednostavnosti
Matrix::Matrix(const float x) : Matrix({{ x }}) { }

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
      throw std::invalid_argument("the given matrix is invalid!");
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

// getteri
const size_t Matrix::getN() {
  return this->N;
}
const size_t Matrix::getM() {
  return this->M;
}

std::string Matrix::toString() const {
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

inline cl_kernel Matrix::loadKernel(const Matrix &other, const std::string &name) const {
  std::unordered_map<std::string, cl_kernel> *kernels;
  if(other.M == this->M && other.N == this->N) {
    kernels = &basicKernels;
    if(kernels->find(name) == kernels->end()) {
      cl_kernel new_kernel;
      buildIfNeeded(&basicOpsProgram, &new_kernel, name.c_str(), srcCode, srcLen);
      (*kernels)[name] = new_kernel;
    }
  }
  else if(other.N == 1 && other.M == 1) {
    kernels = &scalarKernels;
    if(kernels->find(name) == kernels->end()) {
      cl_kernel new_kernel;
      buildIfNeeded(&scalarOpsProgram, &new_kernel, name.c_str(), scalarSrcCode, scalarSrcLen);
      (*kernels)[name] = new_kernel;
    }
  }
  else if(this->N == 1 && this->M == 1) {
    // TODO: skalarano u drugom smjeru...
  }
  else {
    throw std::logic_error("invalid matrix dimensions!");
  }

  return (*kernels)[name];
}

Matrix Matrix::operator+(const Matrix &other) const {
  int _err;
  cl_mem out_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, this->N * this->M * sizeof(float), nullptr, &_err);
  checkError(_err);
  
  cl_kernel addKernel = this->loadKernel(other, "add");

  checkError(clSetKernelArg(addKernel, 0, sizeof(float *), &this->data->data));
  checkError(clSetKernelArg(addKernel, 1, sizeof(float *), &other.data->data));
  checkError(clSetKernelArg(addKernel, 2, sizeof(float *), &out_buffer));
  checkError(clSetKernelArg(addKernel, 3, sizeof(const int), &this->M));

  const size_t global_work_size[] = { this->N, this->M };
  checkError(_err);
  checkError(clEnqueueNDRangeKernel(globalQueue, addKernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  return Matrix(out_buffer, this->N, this->M);
}

Matrix Matrix::operator-(const Matrix &other) const {
  int _err;
  cl_mem out_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, this->N * this->M * sizeof(float), nullptr, &_err);
  checkError(_err);

  cl_kernel subKernel = this->loadKernel(other, "sub");

  clSetKernelArg(subKernel, 0, sizeof(float *), &this->data->data);
  clSetKernelArg(subKernel, 1, sizeof(float *), &other.data->data);
  clSetKernelArg(subKernel, 2, sizeof(float *), &out_buffer);
  clSetKernelArg(subKernel, 3, sizeof(const int), &this->M);

  const size_t global_work_size[] = { this->N, this->M };
  checkError(clEnqueueNDRangeKernel(globalQueue, subKernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  return Matrix(out_buffer, this->N, this->M);
}

Matrix Matrix::operator-() const {
  int _err;
  cl_mem out_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, this->N * this->M * sizeof(float), nullptr, &_err);
  checkError(_err);

  static cl_kernel negateKernel;
  buildIfNeeded(&basicOpsProgram, &negateKernel, "matrixNegate", srcCode, srcLen);

  clSetKernelArg(negateKernel, 0, sizeof(float *), &this->data->data);
  clSetKernelArg(negateKernel, 1, sizeof(float *), &out_buffer);
  clSetKernelArg(negateKernel, 2, sizeof(const int), &this->M);

  const size_t global_work_size[] = { this->N, this->M };
  checkError(clEnqueueNDRangeKernel(globalQueue, negateKernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  return Matrix(out_buffer, this->N, this->M);
}

Matrix Matrix::operator*(const Matrix &other) const {
  int _err;
  cl_mem out_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, this->N * this->M * sizeof(float), nullptr, &_err);
  checkError(_err);

  cl_kernel mulKernel = this->loadKernel(other, "mul");

  clSetKernelArg(mulKernel, 0, sizeof(float *), &this->data->data);
  clSetKernelArg(mulKernel, 1, sizeof(float *), &other.data->data);
  clSetKernelArg(mulKernel, 2, sizeof(float *), &out_buffer);
  clSetKernelArg(mulKernel, 3, sizeof(const int), &this->M);

  const size_t global_work_size[] = { this->N, this->M };
  checkError(clEnqueueNDRangeKernel(globalQueue, mulKernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

  return Matrix(out_buffer, this->N, this->M);
}

Matrix Matrix::operator/(const Matrix &other) const {
  int _err;
  cl_mem out_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, this->N * this->M * sizeof(float), nullptr, &_err);
  checkError(_err);

  cl_kernel divKernel = this->loadKernel(other, "div");

  clSetKernelArg(divKernel, 0, sizeof(float *), &this->data->data);
  clSetKernelArg(divKernel, 1, sizeof(float *), &other.data->data);
  clSetKernelArg(divKernel, 2, sizeof(float *), &out_buffer);
  clSetKernelArg(divKernel, 3, sizeof(const int), &this->M);

  return Matrix(out_buffer, this->N, this->M);
}
