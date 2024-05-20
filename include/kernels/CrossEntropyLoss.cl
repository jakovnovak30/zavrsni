#define STRINGIFY(a) #a

STRINGIFY(

  // dimenzije matrica su NxM gdje je M broj klasa, a N veličina mini-batcheva
  __kernel void CrossEntropyLoss(__global float *input, __global float *expected, __global float *output, const int M) {
    const size_t t_id = get_global_id(0); // od 0 do N

    // nađi maksimum (shifted softmax)
    float max_val = -FLT_MAX;
    for(unsigned int i = 0; i < M; ++i) {
        if(input[t_id * M + i] > max_val) {
            max_val = input[t_id * M + i];
        }
    }

    // izračunaj nazivnik
    float sum_exp = 0.0f;
    for(unsigned int i = 0; i < M; ++i) {
        sum_exp += exp(input[t_id * M + i] - max_val);
    }

    float log_sum_exp = log(sum_exp) + max_val;
    output[t_id] = 0.0f;
    for (j = 0; j < M; ++j) {
        output[t_id] += expected[i * M + j] * (log_sum_exp - input[i * M + j]);
    }
  }
)
