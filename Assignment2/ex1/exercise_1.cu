#include <stdio.h>
#define N 1
#define TPB 256

__global__ void mainKernel()
{
  printf("Hello world! My threadId is %d\n", threadIdx.x);
}

int main()
{

  // Launch kernel to compute and store distance values
  mainKernel<<<N, TPB>>>();
  cudaDeviceSynchronize();
  return 0;
}
