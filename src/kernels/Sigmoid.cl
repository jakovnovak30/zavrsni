#define STRINGIFY(a) #a

STRINGIFY(
  // na ulaz dolaze matrice NxM
  __kernel void sigmoidForward(__global float *input, __global float *output, const int m) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    output[ty * m + tx] = 1 / (1 + exp(-input[ty * m + tx]));
  }

  // na ulaz dolaze matrice NxM
  // delta(gubitak) / delta(aktivacija) -> "izlazniGradijent", dobivamo ga od prethodnog sloja (backpropagation)
  // delta(aktivacija) / delta(ulaz) je derivacija sigmoidalne funkcije za podatke koje smo u koraku unaprijedne propagacije izracunali -> "forwardData"
  // delta(aktivacija) / delta(ulaz) * delta(gubitak) / delta(aktivacija) = delta(gubitak) / delta(ulaz) -> "ulazniGradijent" (računamo ga)
  __kernel void sigmoidBackward(__global float *izlazniGradijent, __global float *forwardData, __global float *ulazniGradijent, const int m) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    // derivacija je f(x) * (1 - f(x)), a forwardData sadrzi vec izracunate vrijednosti f(x)
    ulazniGradijent[ty * m + tx] = izlazniGradijent[ty * m + tx];
    ulazniGradijent[ty * m + tx] *= forwardData[ty * m + tx] * (1 - forwardData[ty * m + tx]);
  }
)
