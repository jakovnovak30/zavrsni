#define STRINGIFY(a) #a

STRINGIFY(
  // na ulaz dolaze matrice NxM
  __kernel void sigmoidForward(__global float *input, __global float *output, const int m) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    if(input[ty * m + tx] >= 0)
      output[ty * m + tx] = 1 / (1 + exp(-input[ty * m + tx]));
    else
      output[ty * m + tx] = exp(input[ty * m + tx]) / (1 + input[ty * m + tx]);
  }
)
