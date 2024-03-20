#include "expression.hpp"
#include <memory>

namespace autograd {
  template <typename T>
  std::shared_ptr<Expression<T>> createVariable(const T &value, const std::string &name) {
    return std::static_pointer_cast<Expression<T>>(std::make_shared<Variable<T>>(value, name));
  }

  template <typename T, typename U>
  std::shared_ptr<Expression<T>> mathFunc(const U &operation) {
    return std::static_pointer_cast<Expression<T>>(std::make_shared<U>(operation));
  }
}
