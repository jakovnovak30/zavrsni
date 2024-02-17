#include "ILossFunc.h"

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
