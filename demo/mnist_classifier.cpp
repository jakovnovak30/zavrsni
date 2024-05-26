#include "IOptimizer.h"
#include "Module.h"
#include "autograd_core/expression.hpp"
#include "autograd_core/autograd_util.hpp"
#include "autograd_core/visualize.hpp"

#include "layers/Linear.h"
#include "layers/Sigmoid.h"

#include "optimizers/SGD.h"
#include "loss_functions/CrossEntropyLossWithSoftmax.h"

#include "data/IDataset.h"
#include "data/Dataloader.h"
#include "data/impl/MNIST.h"

#include "Util.h"
#include <cstdlib>
#include <memory>
#include <iostream>

class MnistClassifier : public Module {
private:
  std::unique_ptr<Linear> layers[3];

public:
  MnistClassifier() {
    this->layers[0] = std::make_unique<Linear>(28*28, 128, false);
    this->layers[1] = std::make_unique<Linear>(128, 64, false);
    this->layers[2] = std::make_unique<Linear>(64, 10, false);
  }

  virtual std::shared_ptr<autograd::Expression<Matrix>> forward(std::shared_ptr<autograd::Expression<Matrix>> ulaz) {
    for(int i=0;i < 3;i++) {
      ulaz = this->layers[i]->forward(ulaz);
      ulaz = std::make_shared<Sigmoid>(ulaz);
    }

    return ulaz;
  }
};

int main() {
  initCL_nvidia();

  std::unique_ptr<IDataset> trainset = std::make_unique<MNIST>("/home/jakov/faks/zavrsni/datasets/MNIST/raw/", true, false);
  std::unique_ptr<IDataset> testset = std::make_unique<MNIST>("/home/jakov/faks/zavrsni/datasets/MNIST/raw/", true, true);

  std::unique_ptr<Dataloader> train_loader = std::make_unique<Dataloader>(*trainset, 10);
  std::unique_ptr<Dataloader> test_loader = std::make_unique<Dataloader>(*trainset, 10);

  std::unique_ptr<MnistClassifier> classifier = std::make_unique<MnistClassifier>();
  std::shared_ptr<IOptimizer> sgd_optim = std::make_shared<SGD>(0.5f);

  int counter = 0;
  while(train_loader->hasNext()) {
    auto [input, labels] = train_loader->nextBatch();
    auto logits = classifier->forward(autograd::createVariable(input, "X"));

    auto loss = std::make_shared<CrossEntropyLossWithSoftmax>(logits, autograd::createVariable(labels, "y", false));
    classifier->backward(loss, sgd_optim);

    if(counter % 10000) {
      std::cout << "batch-logits: " << logits->getValue().toString() << std::endl;
      std::cout << "loss: " << loss->getValue().toString() << std::endl;
      exit(0);
    }

    counter++;
  }

  freeCL();
}
