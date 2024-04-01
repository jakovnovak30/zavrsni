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

/**
 * @class Module
 * @brief skup funkcija koje se zajedno izvršavaju te mogu optimirati, potrebno je jedino nadjačati metodu forward
 * @author Jakov Novak
 */
class Module {
  public:
    /**
     * @brief defaultni konstruktor
     */
    Module() = default;
    /**
     * @brief defaultni virtualni konstruktor
     */
    virtual ~Module() = default;
    
    /**
     * @brief metoda gradi graf funkcije na temelju ulaza koji se može kasnije iskoristiti za optimiranje parametara modula
     *
     * @param ulaz izraz kojeg dovodimo na ulaz modula
     * @throws std::logic_error ukoliko dimenzije ulazne matrice nisu u skladu s definiranim dimenzijama modula
     */
    virtual std::shared_ptr<autograd::Expression<Matrix>> forward(std::shared_ptr<autograd::Expression<Matrix>> ulaz) = 0;

    /**
     * @brief metoda koja na temelju izraza i optimizatora optimira parametre modula, u vecini slucajeva je kompozicija this->forward() i neke funkcije gubitka
     *
     * @param expr izraz na temelju kojeg racunamo gradijente parametara
     * @param optim optimizator koji ažurira parametre modula
     */
    static void backward(std::shared_ptr<autograd::Expression<Matrix>> expr,  std::weak_ptr<IOptimizer> optim);
};
