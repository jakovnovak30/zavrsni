/**
 * @file
 * @brief definicija SGD optimizatora
 * @author Jakov Novak
 */

#include "IOptimizer.h"
#include <CL/cl.h>

/**
 * @class SGD
 * @brief implementacija stohastičkog gradijentnog spusta: \f$ w \coloneq  w - \eta \nabla Q(w) \f$
 */
class SGD : public IOptimizer {
  private:
    /**
     * @brief faktor učenja (eng. learning rate), \f$ \eta \f$
     */
    float learning_rate;
    
    /**
     * @brief OpenCL program koji se koristi za optimizaciju
     */
    cl_program program;
    /**
     * @brief OpenCL jezgra koja se koristi za optimizaciju
     */
    cl_kernel optimize_kernel;

  public:
    /**
     * @brief konstruktor koji prima faktor učenja kao argument
     *
     * @param learning_rate faktor učenja, \f$ \eta \f$
     */
    SGD(float learning_rate);
    
    /**
     * @brief implementacija destruktora koja oslobađa OpenCL objekte
     */
    virtual ~SGD() override;
    
    void optimize(Matrix &layer_parameters, Matrix &layer_gradients) override final;

    void step() override final;
};
