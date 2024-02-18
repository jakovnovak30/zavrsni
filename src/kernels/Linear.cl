#define STRINGIFY(a) #a

STRINGIFY(

    // matrice su tipa NxIN i OUTxIN, a rezultat mora biti NxOUT
  __kernel void matrixMultiplyTransposed(__global float *input, __global float *parameters, __global float *output,
                                          const int n, const int in, const int out) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do OUT

    float retval = 0;
    for(int i=0;i < in;i++) {
      retval = mad(input[ty * in + i], parameters[tx * in + i], retval);
    }

    output[ty * out + tx] = retval;
  }

  // racunaj gradijente tezina (weights)
  // ulaz: outputGrad (matrica NxOUT) -> delta(gubitak) / delta(output)
  // ulaz: forwardData -> ulazni podaci za sloj tijekom unaprijedne propagacije (matrica NxIN)
  // racunamo weightGrad (matrica OUTxIN), gradijente delta(gubitak) / delta(tezina)
  // delta(gubitak) / delta(tezina) = delta(gubitak) / delta(output) * delta(output) / delta(tezina)
  // delta(output) / delta(tezina) = input (spremljeno u matrici "forwardData")
  __kernel void avgWeightGrad(__global float *outputGrad, __global float *weightGrad, __global float *forwardData,
                              const int in, const int out, const int N) {
    const size_t tin = get_global_id(0); // od 0 do IN
    const size_t tout = get_global_id(1); // od 0 do OUT

    // racunamo gradijente za N uzoraka
    float grad_total = 0.0f;
    for(int i = 0;i < N;i++) {
      grad_total += outputGrad[i * out + tout] * forwardData[i * in + tin];
    }

    // spremi prosjecni gradijent za trenutni mini-batch
    weightGrad[tout * in + tin] = grad_total / N;
  }

  // racunaj ulazne gradijente
  // ulaz: outputGrad (matrica NxOUT) -> delta(gubitak) / delta(output)
  // ulaz: forwardData -> ulazni podaci za sloj tijekom unaprijedne propagacije
  // izlaz: inputGrad (matrica NxIN)
  __kernel void inputGrad(__global float *outputGrad, __global float *forwardData, __global float *inputGrad,
                           const int in, const int out) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do IN

    // delta(gubitak) / delta(input) = delta(gubitak) / delta(output) * delta(output) / delta(input)
    // delta(output) / delta(input) = zbroj gubitka po svim tezinama (vidi formulu gore)
    inputGrad[ty * in + tx] = 0.0f;
    for(int i=0;i < out;i++) {
      inputGrad[ty * in + tx] += outputGrad[ty * out + i] * forwardData[ty * in + tx];
    }
  }

   // matrica tipa NxOUT i vektor velicine OUTx1
  __kernel void addBias(__global float *matrix, __global float *bias, const int out) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do OUT

    matrix[ty * out + tx] = matrix[ty * out + tx] + bias[tx];
  }

  // ulaz: NxOUT matrica izlaznog gradijenta
  // izlaz: biasGrad (matrica OUTx1)
  // racuna se delta(gubitak) / delta(bias) koji je jednak izlaznom gradijentu, ali ga svejedno moramo prekopirati
  __kernel void avgBiasGrad(__global float *outputGrad, __global float *biasGrad, const int N, const int out) {
    const size_t t_out = get_global_id(0); // od 0 do OUT

    float grad_total = 0.0f;
    for(int i=0;i < N;i++) {
      grad_total += outputGrad[i * out + t_out];
    }

    biasGrad[t_out] = grad_total / N;
  }
)
