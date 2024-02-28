#include <cmath>
#include <iostream>

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
    Variable(T value) {
      this->value = value;
    }

    void eval() { }
    void derive(T seed) {
      this->partial += seed;
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
  };
}

int main() {
  autograd::Variable x = autograd::Variable<float>(1);
  autograd::Variable y = autograd::Variable<float>(2);
  autograd::Variable z = autograd::Variable<float>(5);

  auto expr = autograd::Exp(-((x + y) * (x + y) * x - z*autograd::Variable<float>(2)));
  // x * (x + y)^2
  expr.eval();

  // derivacija po x je: 2*(x + y) + (x + y)^2 = 2*3 + 9 = 15
  // derivacija po y je: x*2*(x + y) = 2*3 = 6
  expr.autograd::Expression<float>::derive();

  std::cout << expr.value << std::endl;
  std::cout << "po x: " << x.partial << std::endl << "po y: " << y.partial << std::endl;
  std::cout << "po z: " << z.partial << std::endl;

  autograd::Variable k = autograd::Variable<float>(2);
  auto test2 = autograd::Exp(k) * k;

  test2.eval();
  test2.autograd::Expression<float>::derive();

  std::cout << "test2: " << test2.value << std::endl;
  std::cout << "derivacija " << k.partial << std::endl;

  auto x2 = autograd::Variable<float>(5);
  auto y2 = autograd::Variable<float>(3);

  auto test3 = autograd::Variable<float>(1) / (x2 + y2);

  test3.eval(); test3.autograd::Expression<float>::derive();
  std::cout << "test3: " << test3.value << std::endl;
  std::cout << "derivacija po x: " << x2.partial << std::endl;
  std::cout << "derivacija po y: " << y2.partial << std::endl;
}
