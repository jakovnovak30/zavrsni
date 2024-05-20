#define STRINGIFY(a) #a

STRINGIFY(

__kernel void Softmax(__global const float* input, __global float* output, const unsigned int M) {
    // broj trenutnog podatka unutar mini-batcha
    unsigned int t_id = get_global_id(0);

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

    // izračunaj brojnike
    for(unsigned int i = 0; i < M; ++i) {
        output[t_id * M + i] = exp(input[t_id * M + i] - max_val) / sum_exp;
    }
}


)
