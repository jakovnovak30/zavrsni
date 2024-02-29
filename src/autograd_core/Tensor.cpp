#include "Tensor.h"
#include "../Util.h"
#include <CL/cl.h>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>

static cl_program basicOpsProgram = nullptr;
static cl_kernel addKernel = nullptr;
static cl_kernel subKernel = nullptr;
static cl_kernel mulKernel = nullptr;
static const char *srcCode[] =
  {
      #include "../kernels/BasicTensor.cl"
  };
static const size_t srcLen[] = { strlen(srcCode[0]) };

Tensor::Matrix::Matrix(cl_mem data) : data{ data } {
  if(data == nullptr)
    throw std::runtime_error("data must be non-null!");
}

Tensor::Matrix::~Matrix() {
  checkError(clReleaseMemObject(this->data));
}

Tensor::Tensor(cl_mem data, size_t N, size_t M) : N(N), M(M) {
  this->data_matrix = std::make_shared<Matrix>(data);
}

Tensor::Tensor(std::initializer_list<std::initializer_list<float>> mat) {
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

  int _err;
  cl_mem data = clCreateBuffer(globalContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, width * height * sizeof(float), buffer, &_err);
  checkError(_err);

  this->data_matrix = std::make_shared<Matrix>(data);
  this->N = height;
  this->M = width;
  delete[] buffer;
}

std::string Tensor::toString() {
  if(this->N > 100 || this->M > 100) {
    throw std::invalid_argument("tensor is too big to display!");
  }
  std::stringstream ss;

  int _err;
  cl_event event = clCreateUserEvent(globalContext, &_err);
  checkError(_err);
  float *host_ptr = new float[this->N * this->M];
  checkError(clEnqueueReadBuffer(globalQueue, this->data_matrix->data, CL_FALSE, 0, this->N * this->M * sizeof(float), host_ptr, 0, nullptr, &event));
  clWaitForEvents(1, &event);
  clReleaseEvent(event);

  ss << "Tensor(";
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

Tensor Tensor::operator+(const Tensor &other) const {
  int _err;
  cl_mem out_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, this->N * this->M*sizeof(float), nullptr, &_err);
  checkError(_err);

  buildIfNeeded(&basicOpsProgram, &addKernel, "matrixAdd", srcCode, srcLen);

  clSetKernelArg(addKernel, 0, sizeof(float *), &this->data_matrix->data);
  clSetKernelArg(addKernel, 1, sizeof(float *), &other.data_matrix->data);
  clSetKernelArg(addKernel, 2, sizeof(float *), &out_buffer);
  clSetKernelArg(addKernel, 3, sizeof(const int), &this->M);

  const size_t global_work_size[] = { this->N, this->M };
  clEnqueueNDRangeKernel(globalQueue, addKernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);

  return Tensor(out_buffer, this->N, this->M);
}

Tensor Tensor::operator-(const Tensor &other) const {
  int _err;
  cl_mem out_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, this->N * this->M * sizeof(float), nullptr, &_err);
  checkError(_err);

  buildIfNeeded(&basicOpsProgram, &subKernel, "matrixSub", srcCode, srcLen);

  clSetKernelArg(addKernel, 0, sizeof(float *), &this->data_matrix->data);
  clSetKernelArg(addKernel, 1, sizeof(float *), &other.data_matrix->data);
  clSetKernelArg(addKernel, 2, sizeof(float *), &out_buffer);
  clSetKernelArg(addKernel, 3, sizeof(const int), &this->M);

  const size_t global_work_size[] = { this->N, this->M };
  clEnqueueNDRangeKernel(globalQueue, subKernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);

  return Tensor(out_buffer, this->N, this->M);
}

Tensor Tensor::operator*(const Tensor &other) const {
  int _err;
  cl_mem out_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, this->N * this->M * sizeof(float), nullptr, &_err);
  checkError(_err);

  buildIfNeeded(&basicOpsProgram, &mulKernel, "matrixMulScalar", srcCode, srcLen);

  clSetKernelArg(mulKernel, 0, sizeof(float *), &this->data_matrix->data);
  clSetKernelArg(mulKernel, 1, sizeof(float *), &other.data_matrix->data);
  clSetKernelArg(mulKernel, 2, sizeof(float *), &out_buffer);
  clSetKernelArg(mulKernel, 3, sizeof(const int), &this->M);

  const size_t global_work_size[] = { this->N, this->M };
  clEnqueueNDRangeKernel(globalQueue, mulKernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);

  return Tensor(out_buffer, this->N, this->M);
}

Tensor Tensor::operator/(const Tensor &other) const {
  return {{ 1.f }};
}
