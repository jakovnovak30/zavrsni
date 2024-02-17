#include "ILossFunc.h"

// ILossFunc utility functions
cl_context Network::ILayer::getContext(Network &network) {
  return network.context;
}

cl_command_queue Network::ILayer::getQueue(Network &network) {
  return network.queue;
}

cl_device_id Network::ILayer::getDevice(Network &network) {
  return network.device;
}
