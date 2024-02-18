#define STRINGIFY(a) #a

STRINGIFY(

  __kernel void optimizationStep(__global float *parameters, __global float *gradients, const float learning_rate, const int M) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    parameters[ty * M + tx] = parameters[ty * M + tx] - learning_rate * gradients[ty * M + tx];
  }

)
