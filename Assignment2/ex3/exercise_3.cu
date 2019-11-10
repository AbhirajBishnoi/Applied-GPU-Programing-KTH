#include <stdio.h>
#include <sys/time.h>

using namespace std;

struct particle {
    float position[3];
    float velocity[3];
};

struct seed {
    int x;
    int y;
    int z;
};

__host__ __device__ float gen_random(int seed, int particle_id, int iteration,int num_particles)
{
  float rand_num = (seed * particle_id + iteration) % num_particles; 
  return rand_num;
}

__global__ void timestep(struct particle* particles, seed seed, int iteration, int num_particles) {
   
   const int i = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (i < num_particles) {

     // Velocity update:
     particles[i].velocity[0] = gen_random(seed.x, i, iteration, num_particles);
     particles[i].velocity[1] = gen_random(seed.y, i, iteration, num_particles);
     particles[i].velocity[2] = gen_random(seed.z, i, iteration, num_particles);

      // Position update:
      particles[i].position[0] = particles[i].position[0] + particles[i].velocity[0];
      particles[i].position[1] = particles[i].position[1] + particles[i].velocity[1];
      particles[i].position[2] = particles[i].position[2] + particles[i].velocity[2];
   }
}

int main(int argc, char *argv[])
{

  int i = 0, j = 0, num_particles = 10000, num_iterations = 100, tpb = 256, num_blocks = 0;
  struct timeval t0, t1;
  seed seed = {5,6,7};

  for(i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-num_particles") == 0)
    { 
      num_particles = atoi(argv[++i]);
    }
    else if (strcmp(argv[i], "-tpb") == 0)
    {
      tpb = atoi(argv[++i]);
    }
    else if (strcmp(argv[i], "--num_iterations") == 0)
    {
      num_iterations = atoi(argv[++i]);
    }
  }

  num_blocks = (num_particles/tpb + 1); 

  struct particle *particlesCPU = (struct particle*)malloc(num_particles * sizeof(struct particle));
  struct particle *particlesGPU = (struct particle*)malloc(num_particles * sizeof(struct particle));
  struct particle *resultGPUSimulation = (struct particle*)malloc(num_particles * sizeof(struct particle));

  printf("\n\nComputing simulation on the CPU..."); 

  gettimeofday(&t0, 0);

  for(i = 0; i < num_iterations; i++)
  {
     for(j = 0; j < num_particles; j++)
     {
        // Velocity update:
        particlesCPU[j].velocity[0] = gen_random(seed.x, j, i, num_particles);
        particlesCPU[j].velocity[1] = gen_random(seed.y, j, i, num_particles);
        particlesCPU[j].velocity[2] = gen_random(seed.z, j, i, num_particles);

        // Position update:
        particlesCPU[j].position[0] = particlesCPU[j].position[0] + particlesCPU[j].velocity[0];
        particlesCPU[j].position[1] = particlesCPU[j].position[1] + particlesCPU[j].velocity[1];
        particlesCPU[j].position[2] = particlesCPU[j].position[2] + particlesCPU[j].velocity[2];
     }
  }

  gettimeofday(&t1, 0);
  long elapsed_cpu = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;

  printf("Done!");
  printf("\n\nTotal time taken for %d particles and %d iterations on the CPU alone = %ld microseconds", num_particles, num_iterations, elapsed_cpu); 
  cudaMalloc(&particlesGPU, num_particles * sizeof(struct particle));

  gettimeofday(&t0, 0); 
  
  printf("\n\nComputing simulation on the GPU...");

  for (int i = 0; i < num_iterations; i++) {
    timestep<<<num_blocks, tpb>>>(particlesGPU, seed, i, num_particles);
    cudaDeviceSynchronize();
  }

  gettimeofday(&t1, 0);
  long elapsed_gpu = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;

  printf("Done!");
  printf("\n\nTotal time taken for %d particles and %d iterations using the GPU = %ld microseconds", num_particles, num_iterations, elapsed_gpu);
 
  // comparing the results of the two versions

  printf("\n\nComparing the output for each implementation...");

  cudaMemcpy(resultGPUSimulation, particlesGPU, num_particles * sizeof(struct particle), cudaMemcpyDeviceToHost);

  int flag_comparison = 0;
  
  for (i = 0; i < num_particles; i++)
  {
    if( (abs(particlesCPU[i].velocity[0] - resultGPUSimulation[i].velocity[0]) > 1) || (abs(particlesCPU[i].velocity[1] - resultGPUSimulation[i].velocity[1]) > 1) || (abs(particlesCPU[i].velocity[2] - resultGPUSimulation[i].velocity[2]) > 1) || (abs(particlesCPU[i].position[0] - resultGPUSimulation[i].position[0]) > 1)|| (abs(particlesCPU[i].position[1] - resultGPUSimulation[i].position[1]) > 1) || (abs(particlesCPU[i].position[2] - resultGPUSimulation[i].position[2]) > 1) ) 
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


  // clean up
  free(particlesCPU);
  cudaFree(particlesGPU);

  return 0;
}
