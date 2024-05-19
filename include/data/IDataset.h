/**
 * @file
 * @brief apstraktni razred koji predstavlja nekakav skup podataka
 *
 * @author Jakov Novak
 */

#pragma once
#include "autograd_core/Matrix.h"

/**
 * @class Dataset
 * @brief razred koji predstavlja sučelje za bilo koji skup podataka koji se može učitati uz pomoć Dataloader razreda
 */
class IDataset {
public:
  IDataset() = default;
  virtual ~IDataset() = default;

  /**
   * @brief vraća veličinu pojedinog elementa u skupu podataka (u bajtovima)
   *
   * @return
   */
  virtual size_t getElementSize() = 0;
  /**
   * @brief vraća veličinu skupa podataka (broj elemenata)
   *
   * @return
   */
  virtual size_t getSize() = 0;
  /**
   * @brief funkcija pristupa pojedinom podatku
   *
   * @param index indeks podatka, od 0 do this->getSize(), ne uključujući
   * @return vraća podatake na zadanom indeksu, zajedno s njihovim indeksom
   */
  virtual std::pair<Matrix, Matrix> operator[](size_t index) = 0;
};
