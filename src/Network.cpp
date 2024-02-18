#include "Network.h"
#include "IOptimizer.h"
#include "Util.h"
#include <CL/cl.h>

#ifdef DEBUG
#include <iostream>
#endif

static int _err;

Network::Network(cl_context context, cl_device_id device, std::list<ILayer *> layers) : context{ context }, device{ device }, layers{ layers }
{
  this->queue = clCreateCommandQueueWithProperties(context, device, nullptr, &_err);
  checkError(_err);
}

Network::~Network() {
  for(ILayer *layer : this->layers) {
    delete layer;
  }
  checkError(clReleaseCommandQueue(queue));
}

cl_mem Network::forward(void *input_buffer, const size_t N, const size_t M) {
  cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, N * M, nullptr, &_err);
  checkError(_err);

  checkError(clEnqueueWriteBuffer(queue, device_buffer, CL_TRUE, 0, N * M * sizeof(float), input_buffer, 0, nullptr, nullptr));

  return this->forward({ device_buffer, N, M });
}

cl_mem Network::forward(cl_mem input_buffer, const size_t N, const size_t M) {
  return this->forward({ input_buffer, N, M });
}

cl_mem Network::forward(Matrix input_matrix) {
  Matrix current_matrix = input_matrix;

  for(ILayer *layer : this->layers) {
    current_matrix = layer->forward(*this, current_matrix);
  }

  // vrati referencu na izlazni sloj
  #ifdef DEBUG
  std::cout << "[DEBUG]: dimenzije izlaznog sloja: " << current_matrix.N << "x" << current_matrix.M << std::endl;
  #endif
  return current_matrix.data;
}

void Network::backward(Matrix &probs, Matrix &expected, ILossFunc *loss_func, IOptimizer *optim) {
  Matrix output_grad = loss_func->calculate_gradient(*this, probs, expected);

  // izracunaj gradijente
  for(auto it=this->layers.rbegin();it != this->layers.rend();it++) {
    output_grad = (*it)->backward(*this, output_grad);
  }

  // pozivi optimizatora
  for(const ILayer *layer : this->layers) {
    //optim->optimize(layer->params, layer->gradients);
  }
}

// ILayer utility functions
cl_context Network::ILayer::getContext(Network &network) {
  return network.context;
}

cl_command_queue Network::ILayer::getQueue(Network &network) {
  return network.queue;
}

cl_device_id Network::ILayer::getDevice(Network &network) {
  return network.device;
}
