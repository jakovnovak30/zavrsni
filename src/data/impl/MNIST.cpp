/**
 * @file
 * @brief implementacija MNIST skupa podataka
 */

#include <CL/cl.h>
#include <cassert>
#include <vector>
#include <fstream>
#include <stdexcept>
#include "data/impl/MNIST.h"
#include "Util.h"

MNIST::MNIST(const std::string &path, bool normalize, bool test) : normalize(normalize) {
  if(!test) {
    this->data.open(path + "/train-images.idx3-ubyte", std::ios::binary);
    this->labels.open(path + "/train-labels.idx1-ubyte", std::ios::binary);
  }
  else {
    this->data.open(path + "/t10k-images.idx3-ubyte", std::ios::binary);
    this->labels.open(path + "/t10k-labels.idx1-ubyte", std::ios::binary);
  }

  if(!this->data || !this->labels) {
    throw std::runtime_error("Error opening files!");
  }

  unsigned int magicnum_labels, magicnum_data;
  this->labels.read(reinterpret_cast<char *>(&magicnum_labels), 4);
  this->data.read(reinterpret_cast<char *>(&magicnum_data), 4);

  magicnum_labels = __builtin_bswap32(magicnum_labels);
  magicnum_data = __builtin_bswap32(magicnum_data);

  if(magicnum_labels != 2049 || magicnum_data != 2051) {
    throw std::logic_error("Wrong path. Files aren't part of MNIST dataset");
  }

  unsigned int num_items, num_items_data;
  this->labels.read(reinterpret_cast<char *>(&num_items), 4);
  num_items = __builtin_bswap32(num_items);

  this->data.read(reinterpret_cast<char *>(&num_items_data), 4);
  num_items_data = __builtin_bswap32(num_items_data);
  unsigned int row_count, col_count;
  this->data.read(reinterpret_cast<char *>(&row_count), 4);
  this->data.read(reinterpret_cast<char *>(&col_count), 4);
  row_count = __builtin_bswap32(row_count); col_count = __builtin_bswap32(col_count);

  assert(num_items == num_items_data);
  assert(row_count == 28 && col_count == 28);
  this->size = num_items;
}

size_t MNIST::getElementSize() {
  return 28 * 28 * sizeof(float);
}

size_t MNIST::getLabelDims() {
  return 10;
}

size_t MNIST::getSize() {
  return this->size;
}

std::pair<Matrix, Matrix> MNIST::operator[](size_t index) {
  if(index > this->getSize()) {
    throw std::runtime_error("Out of bounds!");
  }

  this->data.seekg(16 + index * 28 * 28);
  this->labels.seekg(8 + index);

  float *data_buffer = new float[28*28];
  float label;
  unsigned char raw_label;
  this->labels.read(reinterpret_cast<char *>(&raw_label), 1);
  label = static_cast<float>(raw_label);
  std::vector<float> lista(10);
  for(int i=0;i < 10;i++) {
    if(i == label)
      lista[i] = 1;
    else
      lista[i] = 0;
  }

  for(int i=0;i < 28*28;i++) {
    unsigned char curr_val;
    this->data.read(reinterpret_cast<char *>(&curr_val), 1);

    if(this->normalize)
      data_buffer[i] = static_cast<float>(curr_val) / 255;
    else
      data_buffer[i] = static_cast<float>(curr_val);
  }

  int _err;
  cl_mem data_cl = clCreateBuffer(globalContext, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, 28*28*sizeof(float), data_buffer, &_err);
  checkError(_err);

  return { Matrix(data_cl, 1, 28*28), Matrix({ lista }) };
}
