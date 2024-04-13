#define STRINGIFY(a) #a

STRINGIFY(
  // matrica NxM i skalar
  __kernel void add(__global const float *mat, __global const float *scalar, __global float *out, const int M) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    out[ty * M + tx] = mat[ty * M + tx] + *scalar;
  }

  // matrica NxM i skalar
  __kernel void sub(__global const float *mat, __global const float *scalar, __global float *out, const int M) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    out[ty * M + tx] = mat[ty * M + tx] - *scalar;
  }

  // matrica NxM i skalar
  __kernel void mul(__global const float *mat, __global const float *scalar, __global float *out, const int M) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    out[ty * M + tx] = mat[ty * M + tx] * *scalar;
  }
  // matrica NxM i skalar
  __kernel void div(__global const float *mat, __global const float *scalar, __global float *out, const int M) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    out[ty * M + tx] = mat[ty * M + tx] / *scalar;
  }
)
