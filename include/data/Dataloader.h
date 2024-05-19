/**
 * @file
 * @brief razred koji čita određeni skup podataka
 *
 * @author Jakov Novak
 */
#include <cstddef>
#include "data/IDataset.h"

/**
 * @class Dataloader
 * @brief klasa koja služi za učitivanje podataka iz skupa podataka u minibatchevima
 */
class Dataloader {
  // TODO: dokumentacija
private:
  size_t batch_num, batch_size;
  IDataset &dataset;
public:
  Dataloader(IDataset &dataset, size_t batch_size);

  std::pair<Matrix, Matrix> nextBatch();
};
