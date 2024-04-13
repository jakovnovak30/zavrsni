#include "data/Dataset.h"
#include <stdexcept>

Dataset::iterator::iterator(Dataset &parent, size_t index) : parent(parent), index(index) {
  if (index < 0 || index > parent.getSize())
    throw std::out_of_range("index out of range");
}

Dataset::iterator& Dataset::iterator::operator++() {
  this->index++;
  return *this;
}

Matrix Dataset::iterator::operator*() {
  return parent[this->index];
}

Dataset::iterator Dataset::begin() {
  return Dataset::iterator(*this, 0);
}

Dataset::iterator Dataset::end() {
  return Dataset::iterator(*this, this->getSize());
}
