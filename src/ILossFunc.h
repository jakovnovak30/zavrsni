#pragma once

class ILossFunc;
class Network;
#include "Network.h"
#include "Matrix.h"
#include <list>

class ILossFunc {
protected:
  static cl_context getContext(Network &network);
  static cl_command_queue getQueue(Network &network);
  static cl_device_id getDevice(Network &network);
public:
  // input: dvije matrice tipa NxC, gdje je N veličina "minibatcha", a C broj klasa
  // output: vektor veličine Nx1 koji ima gubitak za svaki ulaz
  virtual Matrix calculate_loss(Network &network, Matrix &input, Matrix &expected) = 0;
  // input: dvije matrice tipa NxC, gdje je N veličina "minibatcha", a C broj klasa
  // output: matrica veličine NxC, koja ima izracunate gradijente za svaku izlaz
  virtual Matrix calculate_gradient(Network &network, Matrix &input, Matrix &expected) = 0;
};