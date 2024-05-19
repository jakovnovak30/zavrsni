/**
 * @file
 * @brief implementacija MNIST skupa podataka
 */

#include <iostream>
#include <fstream>
#include <stdexcept>
#include "data/impl/MNIST.h"

MNIST::MNIST(const std::string &path, bool test) {
  if(!test) {
    this->data.open(path + "/train-images.idx3-ubyte");
    this->labels.open(path + "/train-labels.idx1-ubyte");
  }
  else {
    this->data.open(path + "t10k-images.idx3-ubyte");
    this->labels.open(path + "t10k-images.idx1-ubyte");
  }

  unsigned int magicnum_labels, magicnum_data;
  this->labels >> magicnum_labels;
  this->data >> magicnum_data;

  if(magicnum_labels != 2049 || magicnum_data != 2051) {
    throw std::logic_error("Wrong path. Files aren't part of MNIST dataset");
  }
}
