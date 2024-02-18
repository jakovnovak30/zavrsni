#include "ILossFunc.h"
#include "Util.h"
#include <CL/cl.h>

float ILossFunc::calculate_avg_loss(Network &network, Matrix &input, Matrix &expected) {
  Matrix loss_n = this->calculate_loss(network, input, expected);
  float *loss_host = new float[loss_n.N];
  checkError(clEnqueueReadBuffer(getQueue(network), loss_n.data, CL_FALSE, 0, sizeof(float) * loss_n.N, loss_host, 0, nullptr, nullptr));

  float loss_avg = 0.0f;
  for(size_t i=0;i < loss_n.N;i++) {
    loss_avg += loss_host[i];
  }
  loss_avg /= loss_n.N;

  delete[] loss_host;
  return loss_avg;
}

// ILossFunc utility functions
cl_context ILossFunc::getContext(Network &network) {
  return network.context;
}

cl_command_queue ILossFunc::getQueue(Network &network) {
  return network.queue;
}

cl_device_id ILossFunc::getDevice(Network &network) {
  return network.device;
}
