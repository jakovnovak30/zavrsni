/**
 * @file
 * @brief apstraktni razred koji predstavlja nekakav skup podataka
 *
 * @author Jakov Novak
 */

#include <iterator>
#include "autograd_core/Matrix.h"

/**
 * @class Dataset
 * @brief razred koji predstavlja sučelje za bilo koji skup podataka koji se može učitati uz pomoć Dataloader razreda
 */
class Dataset {
  // TODO: dokumentacija + konkretna implementacija!
public:
  class iterator {
  private:
    Dataset &parent;
    size_t index;
  public:
    iterator(Dataset &parent, size_t index);

    iterator &operator++();
    Matrix operator*();

    // iterator traits
    using difference_type = Matrix;
    using value_type = Matrix;
    using pointer = const Matrix*;
    using reference = const Matrix&;
    using iterator_category = std::forward_iterator_tag;
  };

  iterator begin();
  iterator end();

  Dataset() = default;
  virtual ~Dataset() = default;

  virtual size_t getSize() = 0;
  virtual Matrix operator[](size_t index) = 0;
};
