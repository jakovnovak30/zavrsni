#pragma once

class Network;
#include "IOptimizer.h"
#include "ILossFunc.h"
#include "Matrix.h"

#include <CL/cl.h>
#include <list>

class Network {
  friend class ILossFunc;
  friend class IOptimizer;

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
        virtual std::shared_ptr<Matrix> forward(Network &network, std::shared_ptr<Matrix> input_matrix) = 0;
        // ulaz: gradijenti prijasnjeg sloja (delta(gubitak) / delta(izlaz)) i optimizator
        virtual std::shared_ptr<Matrix> backward(Network &network, std::shared_ptr<Matrix> prev_grad, std::weak_ptr<IOptimizer> optim) = 0;
    };

    // potreban je trenutni opencl kontekst i opencl uredaj te lista pokazivaca na objekte koji predstavljaju slojeve (moraju biti dinamicki alocirani)
    Network(cl_context context, cl_device_id device, std::list<ILayer *> layers);
    // automatski oslobada memoriju za ILayer klase!
    ~Network();
    
    // na ulaz ide matrica oblika NxM gdje M mora biti jednak ulaznim parametrima prvog sloja
    std::shared_ptr<Matrix> forward(std::shared_ptr<Matrix> input_matrix);
    std::shared_ptr<Matrix> forward(cl_mem input_buffer, const size_t N, const size_t M);
    std::shared_ptr<Matrix> forward(void *input_buffer, const size_t N, const size_t M);

    // predaju se vjerojatnosti izlaznih razreda koje je mreža izračunala + očekivani rezultati (one-hot notacija)
    // zajedno s funkcijom gubitka i optimizatorom
    void backward(std::shared_ptr<Matrix> probs, std::shared_ptr<Matrix> expected, std::weak_ptr<ILossFunc> loss_func, std::weak_ptr<IOptimizer> optim);

  private:
    cl_command_queue queue;
    cl_context context;
    cl_device_id device;
    std::list<ILayer *> layers;
};
