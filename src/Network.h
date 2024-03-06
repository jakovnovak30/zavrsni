#pragma once

#include "autograd_core/expression.hpp"
#include "autograd_core/Matrix.h"
#include "IOptimizer.h"

#include <CL/cl.h>
#include <list>

class Network {
  friend class ILossFunc;
  friend class IOptimizer;

  public:
    // defaultni konstruktor
    Network() = default;
    // defaultni virt. destruktor
    virtual ~Network() = default;
    
    // na ulaz ide matrica oblika NxM gdje M mora biti jednak ulaznim parametrima prvog sloja
    virtual std::shared_ptr<autograd::Expression<Matrix>> forward(std::shared_ptr<autograd::Variable<Matrix>> ulaz) = 0;

    // predaje se izraz kojemu treba optimirati varijable, tj. parametre (u vecini slucajeva rezultat poziva "forward" te neke fje gubitka)
    void backward(std::shared_ptr<autograd::Expression<Matrix>> expr,  std::weak_ptr<IOptimizer> optim);

    // obrisi gradijente svakog sloja
    void clear_grad();
};
