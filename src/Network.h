#pragma once

#include "IOptimizer.h"
#include "ILossFunc.h"
#include "Matrix.h"

#include <CL/cl.h>
#include <list>

class Network {
  friend class ILossFunc;

  public:
    class ILayer {
      protected:
        static cl_context getContext(Network &network);
        static cl_command_queue getQueue(Network &network);
        static cl_device_id getDevice(Network &network);
      public:
        virtual ~ILayer() = default;
        // calculate the result of forward propagation for the current layer
        // returns a handle to the memory used for the result, waits internally for the calculation
        virtual Matrix forward(Network &network, Matrix &input_matrix) = 0;
        // ulaz: gradijenti prijasnjeg sloja (delta(gubitak) / delta(izlaz))
        virtual Matrix backward(Network &network, Matrix &prev_grad) = 0;
    };

    // potreban je trenutni opencl kontekst i opencl uredaj te lista pokazivaca na objekte koji predstavljaju slojeve (moraju biti dinamicki alocirani)
    Network(cl_context context, cl_device_id device, std::list<ILayer *> layers);
    // automatski oslobada memoriju za ILayer klase!
    ~Network();
    
    // na ulaz ide matrica oblika NxM gdje M mora biti jednak ulaznim parametrima prvog sloja
    cl_mem forward(Matrix input_matrix);
    cl_mem forward(cl_mem input_buffer, const size_t N, const size_t M);
    cl_mem forward(void *input_buffer, const size_t N, const size_t M);

    // TODO: backprop za linearni sloj + bar jedna IOptimizer implementacija
    void backward(ILossFunc *loss_func, IOptimizer *optim);

  private:
    cl_command_queue queue;
    cl_context context;
    cl_device_id device;
    std::list<ILayer *> layers;
};