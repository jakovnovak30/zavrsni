#include "IOptimizer.h"

// IOptimizer utility functions
cl_context IOptimizer::getContext(Network &network) {
  return network.context;
}

cl_command_queue IOptimizer::getQueue(Network &network) {
  return network.queue;
}

cl_device_id IOptimizer::getDevice(Network &network) {
  return network.device;
}
