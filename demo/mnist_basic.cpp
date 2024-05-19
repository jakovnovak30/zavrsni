#include "data/impl/MNIST.h"
#include "data/Dataloader.h"
#include "Util.h"
#include <iostream>

int main() {
  initCL_nvidia();
  IDataset *dataset = new MNIST("/home/jakov/faks/zavrsni/datasets/MNIST");

  Dataloader loader(*dataset, 5);

  std::pair<Matrix, Matrix> batch = loader.nextBatch();
  std::cout << "data: " << batch.first.toString() << std::endl;
  std::cout << "labels: " << batch.second.toString() << std::endl;
  delete dataset;
  freeCL();
}
