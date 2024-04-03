/**
 * @file
 * @brief definicija sučelja IOptimizer
 * @author Jakov Novak
 */

#pragma once

#include "autograd_core/Matrix.h"
#include <CL/cl.h>

/**
 * @class IOptimizer
 * @brief apstraktna klasa koja definira sučelje za optimizatore
 * @author Jakov Novak
 */
class IOptimizer {
  public:
    /**
     * @brief generički virtualni destruktor kojeg izvedene klase mogu nadjačati
     */
    virtual ~IOptimizer() = default;

    /**
     * @brief funkcija na temelju izračunatih parametara vrši korak optimizacije za zadanu varijablu
     *
     * @param layer_parameters matrica koja pokazuje na memoriju gdje su spremljeni parametri
     * @param layer_gradients matrica koja sadrži informacije o gradijentima
     * @throws std::logic_error ukoliko su dimenzije matrica različite
     */
    virtual void optimize(Matrix &layer_parameters, Matrix &layer_gradients) = 0;
    
    /**
     * @brief opcionalna funkcija kojoj naglasimo nekim optimizatorima da je gotova jedna iteracija radi azuriranja internog stanja
     */
    virtual void step() = 0;
};
