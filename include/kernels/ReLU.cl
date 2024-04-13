#define STRINGIFY(a) #a

STRINGIFY(
  // na ulaz dolazi matrica NxM, koju direktno mijenjamo
  __kernel void reluForward(__global float *input, __global float *output, const int m) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    output[ty * m + tx] = fmax(0, input[ty * m + tx]);
  }
)
