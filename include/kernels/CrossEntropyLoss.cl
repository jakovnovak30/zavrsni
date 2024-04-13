#define STRINGIFY(a) #a

STRINGIFY(

  // dimenzije matrica su NxC gdje je C broj klasa, a N veličina mini-batcheva
  __kernel void calculateLoss(__global float *input, __global float *expected, __global float *output, const int C) {
    const size_t t_id = get_global_id(0); // od 0 do N

    float loss = 0.0f;
    for(int i=0;i < C;i++) {
      const float dobiveno = input[t_id * C + i];
      const float ocekivano = expected[t_id * C + i];

      loss += dobiveno * log(ocekivano) + (1 - ocekivano) * log(1 - dobiveno);
    }

    loss = -loss;
    output[t_id] = loss;
  }
)
