#include <cmath>
#include <graphviz/cgraph.h>
#include <string>

namespace autograd {
  template <typename T>
  struct Add;
  template <typename T>
  struct Sub;
  template <typename T>
  struct Mult;
  template <typename T>
  struct Div;
  template <typename T>
  struct Neg;
  template <typename T>
  struct Variable;

  template <typename T>
  struct Expression {
    T value;

    virtual void eval() = 0;
    virtual void derive(T seed) = 0;
    virtual void addSubgraph(Agraph_t *graph, Agnode_t *prev) const = 0;

    void derive() {
      this->derive(1);
    }

    Add<T> operator+(Expression &&other) const {
      return Add<T>(*this, other);
    }
    Add<T> operator+(Expression &other) {
      return Add<T>(*this, other);
    }

    Sub<T> operator-(Expression &&other) {
      return Sub<T>(*this, other);
    }
    Sub<T> operator-(Expression &other) {
      return Sub<T>(*this, other);
    }

    Mult<T> operator*(Expression &&other) {
      return Mult<T>(*this, other);
    }
    Mult<T> operator*(Expression &other) {
      return Mult<T>(*this, other);
    }

    Div<T> operator/(Expression &&other) {
      return Div<T>(*this, other);
    }
    Div<T> operator/(Expression &other) {
      return Div<T>(*this, other);
    }

    Neg<T> operator-() {
      return Neg<T>(*this);
    }
  };

  template <typename T>
  struct Variable : public Expression<T> {
    T partial = 0;
    std::string name;

    Variable(T value, const std::string &name) {
      this->value = value;
      this->name = name;
    }

    void eval() { }
    void derive(T seed) {
      this->partial += seed;
    }

    void addSubgraph(Agraph_t *graph, Agnode_t *caller) const {
      Agnode_t *curr = agnode(graph, (char *) name.c_str(), 1);
      if(caller != nullptr)
        agedge(graph, curr, caller, nullptr, 1);
    }
  };

  template <typename T>
  struct BinaryOperator : public Expression<T> {
    Expression<T> &left, &right;

    BinaryOperator(Expression<T> &&left, Expression<T> &&right) {
      this->left = std::move(left);
      this->right = std::move(right);
    }
    BinaryOperator(Expression<T> &left, Expression<T> &right) : left(left), right(right) { }
  };

  template <typename T>
  struct UnaryOperator : public Expression<T> {
    Expression<T> &prev;

    UnaryOperator(Expression<T> &&prev) : prev(prev) { }
    UnaryOperator(Expression<T> &prev) : prev(prev) { }
  };

  static int counter_add = 0;
  template <typename T>
  struct Add : public BinaryOperator<T> {
    Add(Expression<T> &&left, Expression<T> &&right) : BinaryOperator<T>(left, right) { }
    Add(Expression<T> &left, Expression<T> &right) : BinaryOperator<T>(left, right) { }

    void eval() {
      this->left.eval(); this->right.eval();
      this->value = this->left.value + this->right.value;
    }
    void derive(T seed) {
      this->left.derive(seed);
      this->right.derive(seed);
    }
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const {
      Agnode_t *curr = agnode(graph, (char *) ("add" + std::to_string(counter_add++)).c_str(), 1);
      agedge(graph, curr, prev, nullptr, 1);

      this->left.addSubgraph(graph, curr);
      this->right.addSubgraph(graph, curr);
    }
  };

  static int counter_sub = 0;
  template <typename T>
  struct Sub : public BinaryOperator<T> {
    Sub(Expression<T> &&left, Expression<T> &&right) : BinaryOperator<T>(left, right) { }
    Sub(Expression<T> &left, Expression<T> &right) : BinaryOperator<T>(left, right) { }

    void eval() {
      this->left.eval(); this->right.eval();
      this->value = this->left.value - this->right.value;
    }
    void derive(T seed) {
      this->left.derive(seed);
      this->right.derive(-seed);
    }
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const {
      Agnode_t *curr = agnode(graph, (char *) ("sub" + std::to_string(counter_sub++)).c_str(), 1);
      agedge(graph, curr, prev, nullptr, 1);

      this->left.addSubgraph(graph, curr);
      this->right.addSubgraph(graph, curr);
    }
  };

  static int counter_mul = 0;
  template <typename T>
  struct Mult : public BinaryOperator<T> {
    Mult(Expression<T> &&left, Expression<T> &&right) : BinaryOperator<T>(left, right) { }
    Mult(Expression<T> &left, Expression<T> &right) : BinaryOperator<T>(left, right) { }

    void eval() {
      this->left.eval(); this->right.eval();
      this->value = this->left.value * this->right.value;
    }
    void derive(T seed) {
      this->left.derive(seed * this->right.value);
      this->right.derive(seed * this->left.value);
    }
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const {
      Agnode_t *curr = agnode(graph, (char *) ("mult" + std::to_string(counter_mul++)).c_str(), 1);
      agedge(graph, curr, prev, nullptr, 1);

      this->left.addSubgraph(graph, curr);
      this->right.addSubgraph(graph, curr);
    }
  };

  static int counter_div = 0;
  template <typename T>
  struct Div : public BinaryOperator<T> {
    Div(Expression<T> &&left, Expression<T> &&right) : BinaryOperator<T>(left, right) { }
    Div(Expression<T> &left, Expression<T> &right) : BinaryOperator<T>(left, right) { }

    void eval() {
      this->left.eval(); this->right.eval();
      this->value = this->left.value / this->right.value;
    }
    void derive(T seed) {
      this->left.derive(seed / this->right.value);
      this->right.derive(seed * (-this->left.value / this->right.value*this->right.value));
    }
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const {
      Agnode_t *curr = agnode(graph, (char *) ("div" + std::to_string(counter_div)).c_str(), 1);
      agedge(graph, curr, prev, nullptr, 1);

      this->left.addSubgraph(graph, curr);
      this->right.addSubgraph(graph, curr);
    }
  };

  static int counter_neg = 0;
  template <typename T>
  struct Neg : public UnaryOperator<T> {
    Neg(Expression<T> &&prev) : UnaryOperator<T>(prev) { }
    Neg(Expression<T> &prev) : UnaryOperator<T>(prev) { }

    void eval() {
      this->prev.eval();
      this->value = - this->prev.value;
    }

    void derive(T seed) {
      this->prev.derive(-seed);
    }
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const {
      Agnode_t *curr = agnode(graph, (char *) ("neg" + std::to_string(counter_neg++)).c_str(), 1);
      agedge(graph, curr, prev, nullptr, 1);

      this->prev.addSubgraph(graph, curr);
    }
  };

  static int counter_exp = 0;
  template <typename T>
  struct Exp : public UnaryOperator<T> {
    Exp(Expression<T> &&prev) : UnaryOperator<T>(prev) { }
    Exp(Expression<T> &prev) : UnaryOperator<T>(prev) { }

    void eval() {
      this->prev.eval();
      this->value = std::exp(this->prev.value);
    }
    void derive(T seed) {
      this->prev.derive(this->value * seed);
    }
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const {
      Agnode_t *curr = agnode(graph, (char *) ("exp" + std::to_string(counter_exp++)).c_str(), 1);
      agedge(graph, curr, prev, nullptr, 1);

      this->prev.addSubgraph(graph, curr);
    }
  };
}
