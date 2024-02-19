#include "Matrix.h"
#include "Util.h"
#include <CL/cl.h>
#include <stdexcept>

Matrix::Matrix(cl_mem data, const size_t N, const size_t M) : data{ data }, N{ N }, M{ M } {
  if(data == nullptr)
    throw std::runtime_error("data must be non-null!");
}

Matrix::~Matrix() {
  checkError(clReleaseMemObject(this->data));
}
