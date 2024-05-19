/**
 * @file
 * @brief razred koji implementira MNIST skup podataka
 */

#include "data/IDataset.h"
#include <fstream>

/**
 * @class MNIST
 * @brief implementacija sučelja "IDataset" za poznati MNIST skup podataka: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
 *
 */
class MNIST : public IDataset {
private:
  std::ifstream data, labels;
public:
  MNIST(const std::string &path, bool test = false);

  virtual size_t getElementSize() override final;

  virtual size_t getSize() override final;

  virtual std::pair<Matrix, Matrix> operator[](size_t index) override final;
};
