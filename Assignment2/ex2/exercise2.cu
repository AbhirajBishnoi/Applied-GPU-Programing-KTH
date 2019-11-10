#include <stdio.h>
#include <iostream>
#include <math.h>

#define TPB 256
#define ARRAY_SIZE 10000
#define N (ARRAY_SIZE/TPB + 1)

using namespace std;

__global__ void saxpy(float *x, float *y, const int a)
{

   const int i = blockIdx.x * blockDim.x + threadIdx.x;

   if (i < ARRAY_SIZE) {
      y[i] = a * x[i] + y[i];
   }
}

int main()
{

  float *x = NULL; // pointer to array of floats on host
  float *y = NULL; // pointer to array of floats on host
  float *result = NULL; // pointer to array that stores the results of SAXPY on the CPU
  float *d_x = NULL; // pointer to array of floats on device 
  float *d_y = NULL; // pointer to array of floats on device
  float *d_result = NULL; // pointer to array that stores the results of SAXPY on the GPU
 
  int i = 0;
  const int a = 3.0; // value of a in a

  // Allocate memory for arrays on CPU 
  x = (float*)malloc(ARRAY_SIZE * sizeof(float));
  y = (float*)malloc(ARRAY_SIZE * sizeof(float));
  result = (float*)malloc(ARRAY_SIZE * sizeof(float));
  d_result = (float*)malloc(ARRAY_SIZE * sizeof(float));
  
  // Allocate memory for arrays on device
  cudaMalloc(&d_x, ARRAY_SIZE * sizeof(float));
  cudaMalloc(&d_y, ARRAY_SIZE * sizeof(float));

  // Initialize with random values on host
  for (int i = 0; i < ARRAY_SIZE; i++) {
    x[i] = rand() % 1000;
    y[i] = rand() % 1000;
  }

  // Copy random values to device
  cudaMemcpy(d_x, x, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

  printf("\nComputing SAXPY on the CPU...");

  for (i = 0; i < ARRAY_SIZE; i++) {
    result[i] = a * x[i] + y[i]; 
  }

  printf("Done!"); 

  printf("\n\nComputing SAXPY on the GPU...");
 
  saxpy<<<N, TPB>>>(d_x, d_y, a);

  printf("Done!");

  // comparing the results of the two versions

  printf("\n\nComparing the output for each implementation...");
 
  cudaMemcpy(d_result, d_y, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

  int flag_comparison = 0; 
  for (i = 0; i < ARRAY_SIZE; i++) {
    if(abs(result[i] - d_result[i]) > 1)
    {
      flag_comparison = 1;
      break;
    }
  }

  if(flag_comparison == 0)
  {
     printf("Correct!");
  }
  else
  {
    printf("Incorrect!");
  }

  free(x);
  free(y);
  cudaFree(d_x);
  cudaFree(d_y);
  return 0;

}
