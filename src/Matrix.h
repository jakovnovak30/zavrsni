#pragma once

#include <CL/cl.h>
#include <array>
#include <memory>
#include <utility>

struct Matrix {
  cl_mem data;
  size_t N, M;
};
