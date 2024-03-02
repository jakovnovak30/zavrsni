#include "expression.hpp"
#include <cmath>
#include <memory>

namespace autograd {
  template <typename T>
  struct Neg;

  template <typename T>
  struct Add : BinaryOperator<T> {
    Add(std::shared_ptr<Expression<T>> left, std::shared_ptr<Expression<T>> right) : BinaryOperator<T>(left, right) { }

    void eval() override {
      this->left->eval(); this->right->eval();
      this->value = this->left->value + this->right->value;
    }
    void _derive(std::shared_ptr<Expression<T>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<T>>> &out_map) override {
      this->left->derive(seed, out_map);
      this->right->derive(seed, out_map);
    }
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const {
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "add");
      agedge(graph, curr, prev, nullptr, 1);

      this->left->addSubgraph(graph, curr);
      this->right->addSubgraph(graph, curr);
    }
  };

  template <typename T>
  struct Sub : BinaryOperator<T> {
    Sub(std::shared_ptr<Expression<T>> left, std::shared_ptr<Expression<T>> right) : BinaryOperator<T>(left, right) { }

    void eval() override {
      this->left->eval(); this->right->eval();
      this->value = this->left->value - this->right->value;
    }
    void _derive(std::shared_ptr<Expression<T>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<T>>> &out_map) override {
      this->left->derive(seed, out_map);
      this->right->derive(std::make_shared<Neg<T>>(seed), out_map);
    }
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override {
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "subtract");
      agedge(graph, curr, prev, nullptr, 1);

      this->left->addSubgraph(graph, curr);
      this->right->addSubgraph(graph, curr);
    }
  };

  template <typename T>
  struct Mult : BinaryOperator<T> {
    Mult(std::shared_ptr<Expression<T>> left, std::shared_ptr<Expression<T>> right) : BinaryOperator<T>(left, right) { }

    void eval() override {
      this->left->eval(); this->right->eval();
      this->value = this->left->value * this->right->value;
    }
    void _derive(std::shared_ptr<Expression<T>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<T>>> &out_map) override {
      this->left->derive(std::make_shared<Mult<T>>(seed, this->right), out_map);
      this->right->derive(std::make_shared<Mult<T>>(seed, this->left), out_map);
    }
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override {
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "multiply");
      agedge(graph, curr, prev, nullptr, 1);

      this->left->addSubgraph(graph, curr);
      this->right->addSubgraph(graph, curr);
    }
  };

  template <typename T>
  struct Div : BinaryOperator<T> {
    Div(std::shared_ptr<Expression<T>> left, std::shared_ptr<Expression<T>> right) : BinaryOperator<T>(left, right) { }

    void eval() override {
      this->left->eval(); this->right->eval();
      this->value = this->left->value / this->right->value;
    }
    void _derive(std::shared_ptr<Expression<T>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<T>>> &out_map) override {
      this->left->derive(std::make_shared<Div<T>>(seed, this->right), out_map);
      this->right->derive(std::make_shared<Mult<T>>(seed, std::make_shared<Div<T>>(
                          std::make_shared<Neg<T>>(this->left), std::make_shared<Mult<T>>(this->right, this->right))), out_map);
    }
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override {
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "divide");
      agedge(graph, curr, prev, nullptr, 1);

      this->left->addSubgraph(graph, curr);
      this->right->addSubgraph(graph, curr);
    }
  };

  template <typename T>
  struct Neg : UnaryOperator<T> {
    Neg(std::shared_ptr<Expression<T>> prev) : UnaryOperator<T>(prev) { }

    void eval() override {
      this->prev->eval();
      this->value = - this->prev->value;
    }

    void _derive(std::shared_ptr<Expression<T>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<T>>> &out_map) override {
      this->prev->derive(std::make_shared<Neg<T>>(seed), out_map);
    }
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override {
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "negate");
      agedge(graph, curr, prev, nullptr, 1);

      this->prev->addSubgraph(graph, curr);
    }
  };

  template <typename T>
  struct Exp : public UnaryOperator<T>, public std::enable_shared_from_this<Exp<T>> {
    Exp(std::shared_ptr<Expression<T>> prev) : UnaryOperator<T>(prev) { }

    void eval() override {
      this->prev->eval();
      this->value = std::exp(this->prev->value);
    }
    void _derive(std::shared_ptr<Expression<T>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<T>>> &out_map) override {
      this->prev->derive(std::make_shared<Mult<T>>(seed, this->shared_from_this()), out_map);
    }
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override {
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "exp");
      agedge(graph, curr, prev, nullptr, 1);

      this->prev->addSubgraph(graph, curr);
    }
  };

  template <typename T>
  std::shared_ptr<Expression<T>> operator+(std::shared_ptr<Expression<T>> lhs, std::shared_ptr<Expression<T>> rhs) {
    return std::make_shared<Add<T>>(lhs, rhs);
  }

  template <typename T>
  std::shared_ptr<Expression<T>> operator-(std::shared_ptr<Expression<T>> lhs, std::shared_ptr<Expression<T>> rhs) {
    return std::make_shared<Sub<T>>(lhs, rhs);
  }

  template <typename T>
  std::shared_ptr<Expression<T>> operator*(std::shared_ptr<Expression<T>> lhs, std::shared_ptr<Expression<T>> rhs) {
    return std::make_shared<Mult<T>>(lhs, rhs);
  }
  
  template <typename T>
  std::shared_ptr<Expression<T>> operator/(std::shared_ptr<Expression<T>> lhs, std::shared_ptr<Expression<T>> rhs) {
    return std::make_shared<Div<T>>(lhs, rhs);
  }

  template <typename T>
  std::shared_ptr<Neg<T>> operator-(std::shared_ptr<Expression<T>> prev) {
    return std::make_shared<Neg<T>>(prev);
  }
};
