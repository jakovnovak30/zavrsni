#define STRINGIFY(a) #a

STRINGIFY(
  // matrice NxM
  __kernel void add(__global const float *mat1, __global const float *mat2, __global float *out, const int M) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    out[ty * M + tx] = mat1[ty * M + tx] + mat2[ty * M + tx];
  }

  // matrice NxM
  __kernel void sub(__global const float *mat1, __global const float *mat2, __global float *out, const int M) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    out[ty * M + tx] = mat1[ty * M + tx] - mat2[ty * M + tx];
  }

  __kernel void mul(__global const float *mat1, __global const float *mat2, __global float *out, const int M) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    out[ty * M + tx] = mat1[ty * M + tx] * mat2[ty * M + tx];
  }

  __kernel void div(__global const float *mat1, __global const float *mat2, __global float *out, const int M) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    out[ty * M + tx] = mat1[ty * M + tx] / mat2[ty * M + tx];
  }

  __kernel void equals(__global const float *mat1, __global const float *mat2, __global bool *out, const int M) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    if (mat1[ty * M + tx] == mat2[ty * M + tx])
      out[ty * M + tx] = true;
    else
      out[ty * M + tx] = false;
  }

  // __kernel void matrixExp(__global const float *mat, __global const float *out, const int M) {
  //   const size_t ty = get_global_id(0);
  //   const size_t tx = get_global_id(1);
  //
  //   out[ty * M + tx] = exp(mat[ty * M + tx]);
  // }
  //
  // __kernel void matrixLog(__global const float *mat, __global const float *out, const int M) {
  //   const size_t ty = get_global_id(0);
  //   const size_t tx = get_global_id(1);
  //
  //   out[ty * M + tx] = log(mat[ty * M + tx]);
  // }
)
