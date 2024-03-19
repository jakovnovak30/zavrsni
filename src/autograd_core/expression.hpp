#pragma once

#include <graphviz/cgraph.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace autograd {
  template <typename T>
  struct Variable;
  template <typename T>
  struct Add;

  template <typename T>
  struct Expression {
    T value;
    bool evaluated = false;
    bool requires_grad = true;

    virtual void eval() = 0;
    // wrapper oko derive metode
    void derive(std::shared_ptr<Expression<T>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<T>>> &out_map) {
      if(!this->requires_grad)
        return;
      
      this->_derive(seed, out_map);
    }
    // svaka varijabla koja ima requires_grad = true dobiva svoj graf pod "partial"
    virtual void _derive(std::shared_ptr<Expression<T>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<T>>> &out_map) = 0;
    virtual std::optional<std::shared_ptr<const Variable<T>>> find_variable(const std::string &name) const = 0;

  public:
    T &getValue() {
      if(evaluated)
        return this->value;
      
      this->eval();
      this->evaluated = true;
      return this->value;
    }
    // vrati mapu gradijenata
    std::unordered_map<std::string, std::shared_ptr<Expression<T>>> grad() {
      auto out_map = std::unordered_map<std::string, std::shared_ptr<Expression<T>>>();
      this->derive(std::make_shared<Variable<T>>(1, "const 1", false), out_map);
      
      return out_map;
    }

    std::shared_ptr<const Variable<T>> operator[](const std::string &name) {
      auto var = find_variable(name);
      if(!var.has_value())
        throw std::logic_error("variable doesn't exist!");

      return var.value();
    }

    // graphviz
    virtual void addSubgraph(Agraph_t *graph, Agnode_t *prev) const = 0;
  };

  template <typename T>
  struct Variable : Expression<T>, public std::enable_shared_from_this<Variable<T>> {
    std::string name;

    Variable(const T &value, const std::string &name, bool requires_grad = true) {
      this->value = value;
      this->evaluated = true;
      this->name = name;
      this->requires_grad = requires_grad;
    }

    void eval() override { }
    void _derive(std::shared_ptr<Expression<T>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<T>>> &out_map) override {
      if(out_map.find(this->name) != out_map.end()) {
        out_map[this->name] = std::make_shared<Add<T>>(out_map[this->name], seed);
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

  static long id_counter = 0;

  template <typename T>
  struct BinaryOperator : Expression<T> {
    std::shared_ptr<Expression<T>> left, right;

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

  template <typename T>
  struct UnaryOperator : Expression<T> {
    std::shared_ptr<Expression<T>> prev;

    UnaryOperator(std::shared_ptr<Expression<T>> prev) : prev{ prev } {
      this->requires_grad = prev->requires_grad;
    }
    
    virtual std::optional<std::shared_ptr<const Variable<T>>> find_variable(const std::string &name) const override {
      return this->prev->find_variable(name);
    }
  };
}
