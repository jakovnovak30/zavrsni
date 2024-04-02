/**
 * @file
 * @brief definicija SGD optimizatora
 * @author Jakov Novak
 */

#include "IOptimizer.h"
#include <CL/cl.h>

/**
 * @class SGD
 * @brief implementacija stohastičkog gradijentnog spusta: \f$ Q(w) = w - \nablaQ(w) \f$
 *
 */
class SGD : public IOptimizer {
  private:
    float learning_rate;
    
    cl_program program;
    cl_kernel optimize_kernel;

  public:
    SGD(float learning_rate);
    ~SGD();
    
  void optimize(Matrix &layer_parameters, Matrix &layer_gradients) override final;
};
