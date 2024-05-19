#include "data/Dataloader.h"
#include "Util.h"
#include <CL/cl.h>
#include <stdexcept>

Dataloader::Dataloader(IDataset &dataset, size_t batch_size)
  : batch_num(0), batch_size(batch_size), dataset(dataset) {}

Matrix Dataloader::nextBatch() {
  if(batch_num * batch_size > dataset.getSize())
    throw std::logic_error("Dataset is empty!");

  int _err;
  cl_mem batch_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, this->dataset.getElementSize() * this->batch_size, nullptr, &_err);
  checkError(_err);
  cl_mem labels_buffer = clCreateBuffer(globalContext, CL_MEM_READ_ONLY, this->batch_size * sizeof(float), nullptr, &_err);
  checkError(_err);

  for(size_t i=0;i < this->batch_size;i++) {
    std::pair<Matrix, Matrix> curr = this->dataset[(i + this->batch_num * this->batch_size) % this->dataset.getSize()];
    checkError(clEnqueueCopyBuffer(globalQueue, curr.first.data->data, batch_buffer, 0, i * this->dataset.getElementSize(), this->dataset.getElementSize(), 0, nullptr, nullptr));

    checkError(clEnqueueCopyBuffer(globalQueue, curr.second.data->data, labels_buffer, 0, i * sizeof(float), sizeof(float), 0, nullptr, nullptr));
  }

  this->batch_num++;

  return Matrix(batch_buffer, this->batch_size, this->dataset.getElementSize() / sizeof(float));
}
