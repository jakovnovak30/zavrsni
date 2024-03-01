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

  static long id_counter = 0;

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
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "add");
      agedge(graph, curr, prev, nullptr, 1);

      this->left.addSubgraph(graph, curr);
      this->right.addSubgraph(graph, curr);
    }
  };

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
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "subtract");
      agedge(graph, curr, prev, nullptr, 1);

      this->left.addSubgraph(graph, curr);
      this->right.addSubgraph(graph, curr);
    }
  };

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
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "multiply");
      agedge(graph, curr, prev, nullptr, 1);

      this->left.addSubgraph(graph, curr);
      this->right.addSubgraph(graph, curr);
    }
  };

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
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "divide");
      agedge(graph, curr, prev, nullptr, 1);

      this->left.addSubgraph(graph, curr);
      this->right.addSubgraph(graph, curr);
    }
  };

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
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "negate");
      agedge(graph, curr, prev, nullptr, 1);

      this->prev.addSubgraph(graph, curr);
    }
  };

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
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "exp");
      agedge(graph, curr, prev, nullptr, 1);

      this->prev.addSubgraph(graph, curr);
    }
  };
}
