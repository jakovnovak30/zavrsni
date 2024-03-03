#include "Matrix.h"
#include "expression.hpp"
#include "../Util.h"

#include <CL/cl.h>
#include <cstddef>
#include <cstring>
#include <memory>

namespace autograd {
  static const char *codeLinear[] = {
                                    #include "../kernels/Linear.cl"
                                  };
  static const size_t lengths[] = { strlen(codeLinear[0]) };
  static cl_program linearProgram = nullptr;
  static cl_kernel multiplyKernel = nullptr;

  struct MatrixMultplyTransposed : BinaryOperator<Matrix> {
    MatrixMultplyTransposed(std::shared_ptr<Expression<Matrix>> left, std::shared_ptr<Expression<Matrix>> right) : BinaryOperator(left, right) { }

    void eval() override {
      this->left->eval(); this->right->eval();
      buildIfNeeded(&linearProgram, &multiplyKernel, "matrixMultplyTransposed", codeLinear, lengths);
      // TODO: dovrsi
    }
    // TODO: odredi pravu derivaciju
    void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override {
      this->left->derive(seed, out_map);
      this->right->derive(seed, out_map);
    }
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override {
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "matrixMultplyTransposed");
      agedge(graph, curr, prev, nullptr, 1);

      this->left->addSubgraph(graph, curr);
      this->right->addSubgraph(graph, curr);
    }
  };

  struct MatrixVectorAdd : BinaryOperator<Matrix> {
    MatrixVectorAdd(std::shared_ptr<Expression<Matrix>> left, std::shared_ptr<Expression<Matrix>> right) : BinaryOperator(left, right) { }
    
    // TODO: implementacija
  };
}
