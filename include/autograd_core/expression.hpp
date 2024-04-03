/**
 * @file
 * @brief definicija i implementacija generičkih funkcija klase autograd::Expression<T> te njezinih neposrednih nasljednika
 * @author Jakov Novak
 */

#pragma once

#include <graphviz/cgraph.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace autograd {
  template <typename T>
  struct BinaryOperator;
  template <typename T>
  struct UnaryOperator;
  template <typename T>
  struct Expression;
  static long id_counter = 0;
};
#include "autograd_core/basic_operations.hpp"

namespace autograd {
  template <typename T>
  struct Variable;

  /**
   * @brief struktura koja predstavlja apstraktni matematički izraz ili varijablu. Generalno ju možemo zamisliti kao čvor u grafu funkcije
   *
   * @tparam T tip varijabli, najčešće float ili Matrix
   */
  template <typename T>
  struct Expression {
    /**
     * @brief izračunata vrijednost izraza
     */
    T value;
    /**
     * @brief zastavica kojom pamtimo je li izraz već izračunat
     */
    bool evaluated = false;
    /**
     * @brief zastavica kojom pamtimo treba li računati parcijalnu derivaciju za taj (pod)izraz
     */
    bool requires_grad = true;

    /**
     * @brief virtualna metoda kojom se definira evaluiranje izraza
     */
    virtual void eval() = 0;
    /**
     * @brief metoda koja provjerava jesmo li već odredili graf derivacije te ovisno o tome zove _derive ili vraća graf u obliku Expression<T>
     *
     * @param seed vrijednost koja se propagira unazad kao parcijalna derivacija
     * @param out_map referenca na mapu parcijalnih derivacija za sve varijable s requires_grad = true
     */
    void derive(std::shared_ptr<Expression<T>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<T>>> &out_map) {
      if(!this->requires_grad)
        return;
      
      this->_derive(seed, out_map);
    }
    /**
     * @brief funkcija koja rekurzivno gradi graf parcijalnih derivacija
     *
     * @param seed vrijednost koja se propagira unazad kao parcijalna derivacija
     * @param out_map referenca na mapu parcijalnih derivacija za sve varijable s requires_grad = true
     */
    virtual void _derive(std::shared_ptr<Expression<T>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<T>>> &out_map) = 0;
    virtual std::optional<std::shared_ptr<const Variable<T>>> find_variable(const std::string &name) const = 0;

  public:
    /**
     * @brief funkcija koja vraća vrijednost trenutnog izraza
     *
     * @return referenca na this->value
     */
    T &getValue() {
      if(evaluated)
        return this->value;
      
      this->eval();
      this->evaluated = true;
      return this->value;
    }
    /**
     * @brief funkcija koja računa gradijent trenutnog izraza
     *
     * @return mapa gradijenta gdje su ključevi imena varijabli, a vrijednosti std::shared_ptr<Expression<T>>
     */
    std::unordered_map<std::string, std::shared_ptr<Expression<T>>> grad() {
      auto out_map = std::unordered_map<std::string, std::shared_ptr<Expression<T>>>();
      this->derive(std::make_shared<Variable<T>>(1, "const 1", false), out_map);
      
      return out_map;
    }

    /**
     * @brief operator koji traži varijable po imenu
     *
     * @param name ime varijable
     * @return std::shared_ptr na varijablu
     * @throws std::logic_error ako varijabla nije nađena u izrazu
     */
    std::shared_ptr<const Variable<T>> operator[](const std::string &name) {
      auto var = find_variable(name);
      if(!var.has_value())
        throw std::logic_error("variable doesn't exist!");

      return var.value();
    }

    /**
     * @brief funkcija koja sluzi dodavanju trenutnog cvora u graf
     * @see visualize.hpp
     *
     * @param graph pokazivac na graf
     * @param prev cvor koji se nalazi na izlazu trenutnog
     */
    virtual void addSubgraph(Agraph_t *graph, Agnode_t *prev) const = 0;
  };

  /**
   * @brief klasa koja implementira jednu varijablu
   */
  template <typename T>
  struct Variable : Expression<T>, public std::enable_shared_from_this<Variable<T>> {
    std::string name;

    /**
     * @brief konstruktor koji prima početnu vrijednost, ime varijable i zastavicu requires_grad
     *
     * @param value početna vrijednost
     * @param name ime varijable
     * @param requires_grad zastavica koja je istinita ako trebamo računati parcijalnu derivaciju po toj varijabli
     */
    Variable(const T &value, const std::string &name, bool requires_grad = true) {
      this->value = value;
      this->evaluated = true;
      this->name = name;
      this->requires_grad = requires_grad;
    }

    void eval() override { }
    void _derive(std::shared_ptr<Expression<T>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<T>>> &out_map) override {
      if(out_map.find(this->name) != out_map.end()) {
        out_map[this->name] = std::static_pointer_cast<Expression<T>>(std::make_shared<Add<T>>(out_map[this->name], seed));
      }
      else {
       out_map[this->name] = seed;
      }
    }

    void addSubgraph(Agraph_t *graph, Agnode_t *caller) const override {
      Agnode_t *curr = agnode(graph, (char *) name.c_str(), 1);
      if(caller != nullptr)
        agedge(graph, curr, caller, nullptr, 1);
    }

    virtual std::optional<std::shared_ptr<const Variable<T>>> find_variable(const std::string &name) const override {
      if(name == this->name)
        return std::optional(this->shared_from_this());
      else
        return std::nullopt;
    }
  };

  /**
   * @brief generička klasa koja implementira binarne operatore: \f$ y = f(a, b) \f$
   */
  template <typename T>
  struct BinaryOperator : Expression<T> {
    std::shared_ptr<Expression<T>> left, right;

    /**
     * @brief konstruktor koji prima dva izraza: \f$ a \f$ i \f$ b \f$
     *
     * @param left lijevi izraz tipa std::shared_ptr<autograd::Expression<T>>, \f$ a \f$
     * @param right desni izraz tipa std::shared_ptr<autograd::Expression<T>>, \f$ a \f$
     */
    BinaryOperator(std::shared_ptr<Expression<T>> left, std::shared_ptr<Expression<T>> right) : left{ left }, right{ right } {
      this->requires_grad = left->requires_grad || right->requires_grad;
    }

    virtual std::optional<std::shared_ptr<const Variable<T>>> find_variable(const std::string &name) const override {
      auto left_side = this->left->find_variable(name);
      auto right_side = this->right->find_variable(name);
      
      if(left_side.has_value())
        return left_side.value();
      else if(right_side.has_value())
        return right_side.value();
      else
        return std::nullopt;
    }
  };

  /**
   * @brief generička klasa koja implementira unarne operatore: \f$ y = f(x) \f$
   */
  template <typename T>
  struct UnaryOperator : Expression<T> {
    std::shared_ptr<Expression<T>> prev;

    /**
     * @brief konstruktor koji prima jedan izraz: \f$ x \f$
     *
     * @param prev izraz tipa std::shared_ptr<Expression<T>>
     */
    UnaryOperator(std::shared_ptr<Expression<T>> prev) : prev{ prev } {
      this->requires_grad = prev->requires_grad;
    }
    
    virtual std::optional<std::shared_ptr<const Variable<T>>> find_variable(const std::string &name) const override {
      return this->prev->find_variable(name);
    }
  };
}
