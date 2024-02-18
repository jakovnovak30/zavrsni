#include "../IOptimizer.h"
#include "../Network.h"
#include <CL/cl.h>

class SGD : public IOptimizer {
  private:
    float learning_rate;
    
    cl_program program;
    cl_kernel optimize_kernel;

  public:
    SGD(float learning_rate);
    ~SGD();
    
  void optimize(Network &network, Matrix layer_parameters, Matrix layer_gradients) override final;
};
