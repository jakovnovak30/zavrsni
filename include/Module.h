/**
 * @file
 * @brief definicija apstraktne klase Module
 * @author Jakov Novak
 */

#pragma once

#include "autograd_core/expression.hpp"
#include "autograd_core/Matrix.h"
#include "IOptimizer.h"

#include <CL/cl.h>
#include <list>

// slično kao pytorchev Module
class Module {
  public:
    // defaultni konstruktor
    Module() = default;
    // defaultni virt. destruktor
    virtual ~Module() = default;
    
    // na ulaz ide matrica oblika NxM gdje M mora biti jednak ulaznim parametrima prvog sloja
    // gradi se graf izraza kojeg optimizator može optimirati kasnije
    virtual std::shared_ptr<autograd::Expression<Matrix>> forward(std::shared_ptr<autograd::Expression<Matrix>> ulaz) = 0;

    // predaje se izraz kojemu treba optimirati varijable, tj. parametre (u vecini slucajeva rezultat poziva "forward" te neke fje gubitka)
    static void backward(std::shared_ptr<autograd::Expression<Matrix>> expr,  std::weak_ptr<IOptimizer> optim);

    // obrisi gradijente svakog sloja
    void clear_grad();
};
