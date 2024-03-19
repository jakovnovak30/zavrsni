#pragma once

#include "autograd_core/Matrix.h"
#include <CL/cl.h>

class IOptimizer {
  public:
    // funkcija je void zato jer direktno mijenja memoriju na kojoj "žive" parametri
    virtual void optimize(Matrix &layer_parameters, Matrix &layer_gradients) = 0;
    
    // opcionalna funkcija kojoj naglasimo nekim optimizatorima da je gotova jedna iteracija
    // radi azuriranja internog stanja
    virtual void step() = 0;
};
