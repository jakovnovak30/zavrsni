#define STRINGIFY(a) #a

STRINGIFY(
  // na ulaz dolazi matrica NxM, koju direktno mijenjamo
  __kernel void reluForward(__global float *input, __global float *output, const int m) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    output[ty * m + tx] = fmax(0, input[ty * m + tx]);
  }

  // na ulaz dolaze matrice NxM
  // delta(gubitak) / delta(aktivacija) -> "izlazniGradijent", dobivamo ga od prethodnog sloja (backpropagation)
  // delta(aktivacija) / delta(ulaz) je derivacija sigmoidalne funkcije za podatke koje smo u koraku unaprijedne propagacije izracunali -> "forwardData"
  // delta(aktivacija) / delta(ulaz) * delta(gubitak) / delta(aktivacija) = delta(gubitak) / delta(ulaz) -> "ulazniGradijent" (računamo ga)
  __kernel void reluBackward(__global float *izlazniGradijent, __global float *forwardData, __global float *ulazniGradijent, const int m) {
    const size_t ty = get_global_id(0); // od 0 do N
    const size_t tx = get_global_id(1); // od 0 do M

    // derivacija je 1 za pozitivne (nula za ostale*), a forwardData sadrzi vec izracunate vrijednosti f(x)
    // *odabrali smo nulu ovdje za granicu vrijednost f(x) = 0 koja nije diferencijabilna !
    if(forwardData[ty * m + tx] <= 0) {
      ulazniGradijent[ty * m + tx] = 0;
    }
    else {
      ulazniGradijent[ty * m + tx] = izlazniGradijent[ty * m + tx];
    }
  }
)
