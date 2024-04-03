/**
 * @file
 * @brief definicija dodatnih korisnih funkcija vezanih uz autograd_core klase
 * @author Jakov Novak
 */

#include "expression.hpp"
#include <memory>

namespace autograd {
  /**
   * @brief pomoćna funkcija koja generira varijablu te vraća std::shared_ptr na istu varijablu
   *
   * @tparam T tip varijable, najčešće float ili Matrix
   * @param value početna vrijednost varijable
   * @param name ime varijable
   * @param requires_grad zastavica koja određuje trebamo li računati parcijalnu derivaciju po toj varijabli, standardno je true
   * @return std::shared_ptr<Expression<T>> koji pokazuje na upravo konstruiranu varijablu
   */
  template <typename T>
  std::shared_ptr<Expression<T>> createVariable(const T &value, const std::string &name, bool requires_grad = true) {
    return std::static_pointer_cast<Expression<T>>(std::make_shared<Variable<T>>(value, name, requires_grad));
  }
}
