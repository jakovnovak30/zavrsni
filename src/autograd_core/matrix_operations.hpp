#include "Matrix.h"
#include "expression.hpp"
#include "../Util.h"

#include <CL/cl.h>
#include <clblast.h>

#include <cstddef>
#include <cstring>
#include <graphviz/cgraph.h>
#include <memory>

namespace autograd {
  struct MatrixMultply : BinaryOperator<Matrix> {
    bool transposeLeft, transposeRight;

    MatrixMultply(std::shared_ptr<Expression<Matrix>> left, std::shared_ptr<Expression<Matrix>> right,
                  bool transposeLeft = false, bool transposeRight = false)
                  : BinaryOperator(left, right), transposeLeft{ transposeLeft }, transposeRight { transposeRight } { }

    void eval() override {
      this->left->getValue(); this->right->getValue();

      size_t rez_n = this->left->getValue().getN(), rez_m = this->right->getValue().getM();
      if(this->transposeLeft)
        rez_n = this->left->getValue().getM();
      if(this->transposeRight)
        rez_m = this->right->getValue().getN();

      int _err;
      this->value = Matrix(clCreateBuffer(globalContext, CL_MEM_READ_WRITE, rez_n * rez_m * sizeof(float), nullptr, &_err), rez_n, rez_m);
      checkError(_err);

      clblast::Gemm(clblast::Layout::kRowMajor,
                    this->transposeLeft ? clblast::Transpose::kYes : clblast::Transpose::kNo,
                    this->transposeRight ? clblast::Transpose::kYes : clblast::Transpose::kNo,
                    this->left->getValue().getN(), this->left->getValue().getM(), this->right->getValue().getN(), 1.f,
                    this->left->getValue().data->data, 0.f, this->left->getValue().getN(),
                    this->right->getValue().data->data, 0.f, this->right->getValue().getN(), 1.f,
                    this->value.data->data, 0.f, this->value.getN(), &globalQueue);
    }

    void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override {
      this->left->derive(std::make_shared<MatrixMultply>(seed, this->right, this->transposeLeft, !this->transposeRight), out_map);
      this->right->derive(std::make_shared<MatrixMultply>(this->left, seed, !this->transposeLeft, this->transposeRight), out_map);
    }

    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override {
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "matrixMultply");
      agedge(graph, curr, prev, nullptr, 1);

      if(this->transposeLeft) {
        Agnode_t *curr_left = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
        agset(curr_left, (char *) "label", "transpose");
        agedge(graph, curr_left, curr, nullptr, 1);
        this->left->addSubgraph(graph, curr_left);
      }
      else {
        this->left->addSubgraph(graph, curr);
      }
      
      if(this->transposeRight) {
        Agnode_t *curr_right = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
        agset(curr_right, (char *) "label", "transpose");
        agedge(graph, curr_right, curr, nullptr, 1);
        this->right->addSubgraph(graph, curr_right);
      }
      else {
        this->right->addSubgraph(graph, curr);
      }
    }
  };

  struct MatrixVectorAdd : BinaryOperator<Matrix> {
    MatrixVectorAdd(std::shared_ptr<Expression<Matrix>> left, std::shared_ptr<Expression<Matrix>> right) : BinaryOperator(left, right) { }
    
    void eval() override {
      this->left->getValue(); this->right->getValue();
      // TODO: opencl jezgra (vjerojatno iz Linear fajla...)
    }
    // TODO: implementacija derivacije... (vjerojatno normalno kak zbrajanje + neki SumRows ili SumColumns (Contract?) čvor)
    void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override {
      this->left->derive(nullptr, out_map);
      this->right->derive(nullptr, out_map);
    }
    
    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override {
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label", "matrixVectorAdd");
      agedge(graph, curr, prev, nullptr, 1);

      this->left->addSubgraph(graph, curr);
      this->right->addSubgraph(graph, curr);
    }
  };
}
