#define STRINGIFY(a) #a

STRINGIFY(
  // matrica NxM i vektor Nx1
  __kernel void vectorSumReduceRows(__global float *input, __global float *output, const int M) {
    const int t_id = get_global_id(0); // od 0 do N

    float sum = 0;
    for(int i=0;i < M;i++) {
      sum += input[tx * M + i];
    }

    output[t_id] = sum;
  }

  // matrica NxM i vektor Mx1
  __kernel void vectorSumReduceColumns(__global float *input, __global float *output, const int N, const int M) {
    const int t_id = get_global_id(0); // od 0 do M

    float sum = 0;
    for(int i=0;i < N;i++) {
      sum += input[i * M + t_id];
    }

    output[t_id] = sum;
  }

   // matrica tipa NxM i vektor velicine Mx1
  __kernel void matrixAddVector(__global float *matrix_in, __global float *vector_in, __global float *matrix_out, const int M) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    matrix_out[ty * M + tx] = matrix_in[ty * M + tx] + vector_in[tx];
  }
)
