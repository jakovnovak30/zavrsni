#define STRINGIFY(a) #a

STRINGIFY(

  // dimenzije matrica su NxC gdje je C broj klasa, a N veličina mini-batcheva
  // work_dim je samo prva dimenzija (N), jer vraćamo vektor kao ukupni gubitak svih razreda
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

  // dimenzije matrica su NxC gdje je C broj klasa, a N veličina mini-batcheva
  // work_dim je NxC te računamo gradijent za svaku klasu pojedinacno (output je isto NxC)
  __kernel void calculateGradient(__global float *input, __global float *expected, __global float *output, const int C) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do C

    // formula za racunanje parcijalne derivacije i-tog elementa vektora gubitka je: -ocekivano_i / dobiveno_i
    output[ty * C + tx] = - expected[ty * C + tx] / input[ty * C + tx] + (1 - expected[ty * C + tx]) / (1 - input[ty * C + tx]);
  }

)
