#include "Network.h"
#include "ILossFunc.h"
#include "IOptimizer.h"
#include "Matrix.h"
#include "Util.h"
#include <CL/cl.h>
#include <memory>

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

std::shared_ptr<Matrix> Network::forward(void *input_buffer, const size_t N, const size_t M) {
  cl_mem device_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, N * M, nullptr, &_err);
  checkError(_err);

  checkError(clEnqueueWriteBuffer(queue, device_buffer, CL_TRUE, 0, N * M * sizeof(float), input_buffer, 0, nullptr, nullptr));

  return this->forward(std::make_shared<Matrix>(device_buffer, N, M));
}

std::shared_ptr<Matrix> Network::forward(cl_mem input_buffer, const size_t N, const size_t M) {
  return this->forward(std::make_shared<Matrix>(input_buffer, N, M));
}

std::shared_ptr<Matrix> Network::forward(std::shared_ptr<Matrix> input_matrix) {
  std::shared_ptr<Matrix> current_matrix = input_matrix;

  for(ILayer *layer : this->layers) {
    current_matrix = layer->forward(*this, current_matrix);
  }

  // vrati referencu na izlazni sloj
  #ifdef DEBUG
  std::cout << "[DEBUG]: dimenzije izlaznog sloja: " << current_matrix->N << "x" << current_matrix->M << std::endl;
  #endif
  return current_matrix;
}

void Network::backward(std::shared_ptr<Matrix> probs, std::shared_ptr<Matrix> expected,
                       std::weak_ptr<ILossFunc> loss_func, std::weak_ptr<IOptimizer> optim) {
  std::shared_ptr<Matrix> output_grad = loss_func.lock()->calculate_gradient(*this, probs, expected);

  // izracunaj gradijente i usput azuriraj parametre (unutar Network::ILayer::backward)
  for(auto it=this->layers.rbegin();it != this->layers.rend();it++) {
    std::shared_ptr<Matrix> next_grad = (*it)->backward(*this, output_grad, optim);
    // checkError(clReleaseMemObject(output_grad.data)); // obrisi vrijednosti koje ne trebamo vise
    output_grad->data = next_grad->data;
    output_grad->N = next_grad->N;
    output_grad->M = next_grad->M;
  }
  // checkError(clReleaseMemObject(output_grad.data));

  return;
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
