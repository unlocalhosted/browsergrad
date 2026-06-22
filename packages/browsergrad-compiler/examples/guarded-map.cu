__global__ void guarded_map(const float* input, float* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = max(input[i], 0.0);
  }
}
