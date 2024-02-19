#include <CL/cl.h>
#include <CL/opencl.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

#include "../src/layers/Layers.h"
#include "../src/loss_functions/LossFunctions.h"
#include "../src/Network.h"
#include "../src/Util.h"
#include "../src/optimizers/Optimizers.h"

static cl_context context;
static std::default_random_engine generator = std::default_random_engine{};

class Random2DGaussian {
  private:
    float min_x, min_y;
    float max_x, max_y;
    std::normal_distribution<float> dist_x, dist_y;

  public:
  Random2DGaussian() {
    this->min_x = 0, this->min_y = 0;
    this->max_y = 10, this->max_y = 10;
    int delta_x = max_x - min_x;
    int delta_y = max_y - min_y;
    
    float mean_x = (float) rand() / RAND_MAX * delta_x + min_x;
    float mean_y = (float) rand() / RAND_MAX * delta_y + min_y;

    float stddev_x = (float) rand() / RAND_MAX * delta_x / 5;
    float stddev_y = (float) rand() / RAND_MAX * delta_y / 5;

    this->dist_x = std::normal_distribution<float>(mean_x, stddev_x);
    this->dist_y = std::normal_distribution<float>(mean_y, stddev_y);
  }

  std::pair<int, int> get_sample() {
    return { this->dist_x(generator), this->dist_y(generator) };
  }
};

// napravi C razlicitih distribucija i N uzoraka iz svake
// vraca par matrice Nx2 s tockama i polje s tocnim labelama klase
std::pair<std::shared_ptr<Matrix>, size_t *> get_samples(const size_t C, const size_t N) {
  std::vector<std::pair<std::pair<float, float>, int>> uzorci = {};

  for(size_t i=0;i < C;i++) {
    Random2DGaussian *rand_gen = new Random2DGaussian();
    for(size_t j=0;j < N;j++) {
      uzorci.push_back({ rand_gen->get_sample(), i }); // oznaci uzorak i klasu usput
    }
    delete rand_gen;
  }

  // promijesaj dataset
  auto rng = std::default_random_engine{};
  std::shuffle(uzorci.begin(), uzorci.end(), generator);

  float *host_ptr = new float[N * C * 2];
  size_t *labele = new size_t[N * C];
  size_t counter = 0;
  for(auto &pair : uzorci) {
    host_ptr[counter * 2 + 0] = pair.first.first;
    host_ptr[counter * 2 + 1] = pair.first.second;
    labele[counter] = pair.second;
    counter++;
  }

  int _err;
  cl_mem out_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * 2, host_ptr, &_err);
  checkError(_err);
  delete[] host_ptr;

  std::shared_ptr<Matrix> out_matrix = std::make_shared<Matrix>(out_buffer, N, 2);

  return { out_matrix, labele };
}

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
  std::cerr << "OpenCL Error (via pfn_notify): " << errinfo << std::endl;
  throw std::runtime_error("OpenCL");
}

int main() {
  cl_device_id devices[2];   
  cl_platform_id platforms[2];
  cl_device_id device_id;
  cl_uint num_devices;
  cl_uint num_platforms;
  checkError(clGetPlatformIDs(2, platforms, &num_platforms));

  for(int i=0;i < num_platforms;i++) {
    checkError(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 1,
            devices, &num_devices));
    char buffer[1024];

    for(int j=0;j < num_devices;j++) {
      checkError(clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
      printf("Pronaden uredaj: %s\n", buffer);
      // nasli smo dobar uredaj
      if(!strcmp("NVIDIA Corporation", buffer)) {
        checkError(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
        printf("Koristim gpu: %s\n", buffer);
        device_id = devices[j];
      }
    }
  }
  
  cl_int _err;
  context = clCreateContext(NULL, 1, &device_id, pfn_notify, NULL, &_err);
  checkError(_err);
  cl_command_queue kju = clCreateCommandQueueWithProperties(context, device_id, nullptr, &_err);
  checkError(_err);

  const size_t br_klasa = 2;
  const size_t N = 100;
  Network mreza(context, device_id, { new Linear(context, 2, 10), new Sigmoid(), new Linear(context, 10, br_klasa), new Sigmoid() });
  std::pair<std::shared_ptr<Matrix>, size_t *> uzorci = get_samples(br_klasa, N);

  std::shared_ptr<ILossFunc> loss_func = std::make_shared<CrossEntropyLoss>();
  // one-hot notacija za ocekivane vrijednosti
  float *ocekivano = new float[N * br_klasa];
  for(int i=0;i < N;i++) {
    for(int j=0;j < br_klasa;j++) {
      if(j == uzorci.second[i])
        ocekivano[i*br_klasa + j] = 1;
      else
        ocekivano[i*br_klasa + j] = 0;
    }
  }
  cl_mem ocekivano_cl = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N*br_klasa * sizeof(float), ocekivano, &_err);
  checkError(_err);
  std::shared_ptr<Matrix> expected = std::make_shared<Matrix>(ocekivano_cl, N, br_klasa);
  delete[] ocekivano;
  // logistička regresija
  float *dobiveno = new float[N * br_klasa];
  for(int epoch=0;epoch < 2000;epoch++) {
    std::shared_ptr<Matrix> izlaz = mreza.forward(uzorci.first);
    checkError(clEnqueueReadBuffer(kju, izlaz->data, CL_TRUE, 0, N*br_klasa*sizeof(float), dobiveno, 0, nullptr, nullptr));


    if(epoch == 1999) {
      for(int i=0;i < N;i++) {
        float suma_exp = 0;
        for(int j=0;j < br_klasa;j++) {
          suma_exp += exp(dobiveno[i*br_klasa + j]);
        }
        std::cout << "relativne tocnosti: ";
        for(int j=0;j < br_klasa;j++) {
          std::cout << exp(dobiveno[i*br_klasa + j]) / suma_exp << " ";
        }
        std::cout << "očekivani razred: " << uzorci.second[i] << std::endl;
      }
    }
    mreza.backward(izlaz, expected, loss_func, std::make_shared<SGD>(0.5f));
    
    if(epoch % 10 == 0) {
      float avg_loss = loss_func->calculate_avg_loss(mreza, izlaz, expected);
      std::cout << "epoch_no: " << epoch << " loss: " << avg_loss << std::endl;
    }
  }
  delete[] dobiveno;

  clReleaseCommandQueue(kju);
  clReleaseDevice(device_id);
  clReleaseContext(context);
  return 0;
}
