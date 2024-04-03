/**
 * @file
 * @brief definicija dodatnih korisnih funkcija vezanih uz autograd_core klase
 * @author Jakov Novak
 */

#include "expression.hpp"
#include <memory>

namespace autograd {
  template <typename T>
  std::shared_ptr<Expression<T>> createVariable(const T &value, const std::string &name, bool requires_grad = true) {
    return std::static_pointer_cast<Expression<T>>(std::make_shared<Variable<T>>(value, name, requires_grad));
  }

  template <typename T, typename U>
  std::shared_ptr<Expression<T>> mathFunc(const U &operation) {
    return std::static_pointer_cast<Expression<T>>(std::make_shared<U>(operation));
  }
}
