/**
 * @file
 * @brief definicija i implementacija autograd operacija koje su specifične za klasu Matrix
 * @author Jakov Novak
 */

#include "Matrix.h"
#include "expression.hpp"
#include "Util.h"

#include <CL/cl.h>
#include <clblast.h>

#include <cstddef>
#include <cstring>
#include <graphviz/cgraph.h>
#include <memory>
#include <stdexcept>

namespace autograd {
  static cl_program program = nullptr;
  static const char *srcCode[] =
    {
        #include "kernels/MatrixOperations.cl"
    };
  static const size_t srcLen[] = { strlen(srcCode[0]) };

  /**
   * @class MatrixMultply
   * @brief klasa koja predstavlja binarni operator matričnog množenja
   *
   */
  struct MatrixMultply : BinaryOperator<Matrix> {
    /**
     * @brief zastavice za transponiranje lijeve i/ili desne strane
     */
    bool transposeLeft, transposeRight;

    /**
     * @brief konstruktor koji prima dva izraza te (opcionalno) njihove zastavice za transponiranje
     *
     * @param left lijeva matrica
     * @param right desna matrica
     * @param transposeLeft zastavica za transponiraje lijeve matrice
     * @param transposeRight zastavica za transponiraje desne matrice
     */
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

  /**
   * @class VectorSumReduction
   * @brief unarni operator redukcije matrice u vektora zbrajanjem po proizvoljnoj osi
   *
   */
  struct VectorSumReduction : UnaryOperator<Matrix> {
    /**
     * @brief os po kojoj želimo reducirati matricu, može biti 0 (zbraja se po x osi) ili 1 (zbraja se po y osi)
     */
    int axis;
    /**
     * @brief konstruktor kojemo zadajemo izraz i os
     *
     * @param prev izraz koji je ulaz naše funkcije
     * @param axis os redukcije, 0 ili 1
     */
    VectorSumReduction(std::shared_ptr<Expression<Matrix>> prev, int axis = 0) : UnaryOperator<Matrix>(prev), axis(axis) { }

    void eval() override {
      static cl_kernel vectorSumRows = nullptr;
      static cl_kernel vectorSumCols = nullptr;

      int _err;
      cl_mem output_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE,
                                            sizeof(float) * (axis == 0 ? prev->getValue().getN() : prev->getValue().getM()),
                                            nullptr, &_err);
      checkError(_err);
      
      if(this->axis == 0) {
        buildIfNeeded(&program, &vectorSumRows, "vectorSumReduceRows", srcCode, srcLen);

        checkError(clSetKernelArg(vectorSumRows, 0, sizeof(float *), &this->prev->getValue().data->data));
        checkError(clSetKernelArg(vectorSumRows, 1, sizeof(float *), &output_buffer));
        int M = this->prev->getValue().getM();
        checkError(clSetKernelArg(vectorSumRows, 2, sizeof(const int), &M));

        size_t global_work_size[] = { this->prev->getValue().getN() };
        checkError(clEnqueueNDRangeKernel(globalQueue, vectorSumRows, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));
      }
      else {
        buildIfNeeded(&program, &vectorSumCols, "vectorSumReduceColumns", srcCode, srcLen);
        int N = this->prev->getValue().getN(), M = this->prev->getValue().getM();
        
        checkError(clSetKernelArg(vectorSumCols, 0, sizeof(float *), &this->prev->getValue().data->data));
        checkError(clSetKernelArg(vectorSumCols, 1, sizeof(float *), &output_buffer));
        checkError(clSetKernelArg(vectorSumCols, 2, sizeof(const int), &N));
        checkError(clSetKernelArg(vectorSumCols, 3, sizeof(const int), &M));

        size_t global_work_size[] = { this->prev->getValue().getM() };
        checkError(clEnqueueNDRangeKernel(globalQueue, vectorSumCols, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));
      }

      this->value = Matrix(output_buffer, axis == 0 ? prev->getValue().getN() : prev->getValue().getM(), 1);
    }
    void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override {
      this->prev->derive(std::make_shared<VectorSumReduction>(seed, axis), out_map);
    }

    void addSubgraph(Agraph_t *graph, Agnode_t *prev) const override {
      Agnode_t *curr = agnode(graph, (char *) std::to_string(id_counter++).c_str(), 1);
      agset(curr, (char *) "label",
            (std::string("vectorSumReduction") + (axis == 0 ? " on x axis" : "on y axis")).c_str());
      agedge(graph, curr, prev, nullptr, 1);
      this->prev->addSubgraph(graph, curr);
    }
  };

  /**
   * @class MatrixVectorAdd
   * @brief binarni operator zbrajanja matrice s vektorom: \f$ A + \vec{b} \f$
   *
   */
  struct MatrixVectorAdd : BinaryOperator<Matrix> {
    /**
     * @brief konstruktor kojemo zadajemo izraze koje treba zbrojiti
     *
     * @param left matrica koju zbrajamo s vektorom, \f$ A \f$
     * @param right vektor kojeg zbrajamo s matricom, \f$ \vec{b} \f$
     */
    MatrixVectorAdd(std::shared_ptr<Expression<Matrix>> left, std::shared_ptr<Expression<Matrix>> right) : BinaryOperator(left, right) { }
    
    void eval() override {
      static cl_kernel matrixVectorAdd = nullptr;  

      this->left->getValue(); this->right->getValue();
      if(this->left->getValue().getM() != this->right->getValue().getN() || this->right->getValue().getM() != 1)
        throw std::logic_error("Invalid matrix/vector dimensions!");

      buildIfNeeded(&program, &matrixVectorAdd, "matrixVectorAdd", srcCode, srcLen);
      
      size_t M = this->left->getValue().getM(), N = this->right->getValue().getN();
      int _err;
      cl_mem out_buffer = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, N * M * sizeof(float), nullptr, &_err);
      checkError(_err);

      checkError(clSetKernelArg(matrixVectorAdd, 0, sizeof(float *), &this->left->getValue().data->data));
      checkError(clSetKernelArg(matrixVectorAdd, 1, sizeof(float *), &this->right->getValue().data->data));
      checkError(clSetKernelArg(matrixVectorAdd, 2, sizeof(float *), &out_buffer));
      checkError(clSetKernelArg(matrixVectorAdd, 3, sizeof(const int), &M));

      size_t global_work_size[] = {N, M };
      checkError(clEnqueueNDRangeKernel(globalQueue, matrixVectorAdd, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr));

      this->value = Matrix(out_buffer, N, M);
    }
    void _derive(std::shared_ptr<Expression<Matrix>> seed, std::unordered_map<std::string, std::shared_ptr<Expression<Matrix>>> &out_map) override {
      this->left->derive(seed, out_map); // za matricu samo prosijedi seed od prije
      this->right->derive(std::make_shared<VectorSumReduction>(seed, 1), out_map); // za vektor zbroji elemente po stupcima
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
