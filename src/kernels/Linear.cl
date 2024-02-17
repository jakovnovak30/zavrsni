#define STRINGIFY(a) #a

STRINGIFY(

  // matrice su tipa NxIN i OUTxIN, a rezultat mora biti NxOUT
__kernel void matrixMultiplyTransposed(__global float *input, __global float *parameters, __global float *output,
                                        const int n, const int in, const int out) {
  size_t ty = get_global_id(0); // od 0 do N
  size_t tx = get_global_id(1); // od 0 do OUT

  float retval = 0;
  for(int i=0;i < in;i++) {
    retval = mad(input[ty * in + i], parameters[tx * in + i], retval);
  }

  output[ty * out + tx] = retval;
}

 // matrica tipa NxOUT i vektor velicine OUTx1
__kernel void addBias(__global float *matrix, __global float *bias, const int out) {
  size_t ty = get_global_id(0); // od 0 do N
  size_t tx = get_global_id(1); // od 0 do OUT

  matrix[ty * out + tx] = matrix[ty * out + tx] + bias[tx];
}

)
