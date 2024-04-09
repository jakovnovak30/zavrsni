/**
 * @file
 * @brief definicija funkcionalnosti za razred Matrix
 * @author Jakov Novak
 */

#pragma once

#include <CL/cl.h>
#include <initializer_list>
#include <memory>

#include "Util.h"

/**
 * @class Matrix
 * @brief klasa koja enkapsulira podatke o matrici koji su spremljeni na povezanom OpenCL uređaju
 *
 */
class Matrix {
private:
  /**
   * @class opencl_data
   * @brief interna struktura koja sprema cl_mem referencu te se koristi isključivo kao std::shared_ptr da bi se automatski oslobodila memorija na uređaju
   */
  struct opencl_data {
    opencl_data(cl_mem data) : data{ data } { }
    opencl_data(const int N, const int M) {
      int _err;
      this->data = clCreateBuffer(globalContext, CL_MEM_READ_WRITE, N * M * sizeof(float), nullptr, &_err);
      checkError(_err);
    }
    ~opencl_data() { if(this->data != nullptr) checkError(clReleaseMemObject(this->data)); }
    cl_mem data;

    operator cl_mem () {
      return this->data;
    }
  };
  /**
   * @brief dimenzije matrice NxM
   */
  size_t N, M;
  /**
   * funkcija koja učita jezgru i postavi argumente za jednu od osnovnih operacija (zbrajanje, množenje, itd.)
   */
  inline cl_kernel loadKernel(const Matrix &other, const std::string &name) const;

public:
  /**
   * @brief konstruktor koji prima OpenCL memoriju i parametre N i M
   *
   * @param data cl_mem referenca na memoriju
   * @param N broj redaka u matrici
   * @param M broj stupaca u matrici
   */
  Matrix(cl_mem data, size_t N, size_t M);
  /**
   * @brief konstruktor koji prima ugniježdenu inicijalizacijsku listu s elementima tipa float
   *
   * @param mat inicjalizacijska lista
   * @throws std::invalid_argument ako dimenzije liste nemaju smisla, npr. {{3, 1, 2}, {1, -1}}
   */
  Matrix(std::initializer_list<std::initializer_list<float>> mat);
  /**
   * @brief "workaround" za korištenje skalara u svijetu matrica, stvara 1x1 matricu
   *
   * @param x vrijednost skalara
   */
  Matrix(const float x);
  /**
   * @brief prazni konstruktor koji se koristi za neinicijalizirane matrice
   */
  Matrix();

  Matrix operator+(const Matrix &other) const;
  Matrix operator-(const Matrix &other) const;
  Matrix operator-() const;
  Matrix operator*(const Matrix &other) const;
  Matrix operator/(const Matrix &other) const;

  bool operator==(const Matrix &other) const;

  /**
   * @brief pretvori matricu u std::string
   *
   * @throws std::invalid_argument ako je matrica veća od 100x100
   */
  std::string toString() const;

  /**
   * @brief getter za broj N
   *
   * @return vraća broj redaka matrice
   */
  size_t getN();
  /**
   * @brief getter za broj M
   *
   * @return vraća broj stupaca matrice
   */
  size_t getM();
  /**
   * @brief referenca na OpenCL podatke
   */
  std::shared_ptr<opencl_data> data;
};
