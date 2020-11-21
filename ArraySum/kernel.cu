#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{ 
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

#define THREADS_PER_BLOCK 1024
#define BOLCKS_PER_GRID 1024 * 64
#define N THREADS_PER_BLOCK * BOLCKS_PER_GRID

#define REAL double

#define START_CPU {\
begin = clock();

#define STOP_CPU \
end = clock();\
time_cpu = end - begin;}

#define START_GPU {\
HANDLE_ERROR(cudaEventCreate(&start));\
HANDLE_ERROR(cudaEventCreate(&stop));\
HANDLE_ERROR(cudaEventRecord(start, 0));

#define STOP_GPU \
HANDLE_ERROR(cudaEventRecord(stop, 0));\
HANDLE_ERROR(cudaEventSynchronize(stop));\
HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));\
HANDLE_ERROR(cudaEventDestroy(start));\
HANDLE_ERROR(cudaEventDestroy(stop));}

REAL sumArrayByCpu(REAL * array)
{
	REAL sum = 0.0;
	for (int i = 0; i < N; ++i)
	{
		sum += array[i];
	}
	return sum;
}

__global__ void sumArrayByGpuGlobal(REAL* result,const REAL* array)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < N)
	{
		atomicAdd(result, array[tid]);
	}
}

__global__ void sumArrayByGpuShared(REAL* result, const REAL* array)
{
	__shared__ REAL block_result;
	block_result = 0.0f;
	__syncthreads();
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < N)
	{
		atomicAdd(&block_result, array[tid]);
		__syncthreads();
		if (threadIdx.x == 0)
		{
			atomicAdd(result, block_result);
		}
	}
}

__global__ void sumArrayByGpuFewAtomic(REAL* result, const REAL* array)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	REAL part_result = 0;
	int grid_size = gridDim.x * blockDim.x;
	while (tid < N)
	{
		part_result += array[tid];
		tid += grid_size;
	}
	atomicAdd(result, part_result);
}




int main()
{
	//use for calculcating cost time
	clock_t begin, end;
	long time_cpu;
	cudaEvent_t start, stop;
	float elapsed_time;

	REAL* h_array = (REAL*)malloc(N * sizeof(REAL));
	if (h_array == NULL)
	{
		printf("memory allocation failed!");
		exit(EXIT_FAILURE);
	}
	REAL* h_result = (REAL*)malloc(sizeof(REAL));
	if (h_result == NULL)
	{
		printf("memory allocation failed!");
		exit(EXIT_FAILURE);
	}

	REAL* d_result;
	REAL* d_array;

	for (int i = 0; i < N; ++i)
	{
		h_array[i] = i / 10000.0f;
	}

	printf("type\t\t\t\tsum\t\tcost_time(ms)\t\tspeedup_ratio\n");
	printf("-------------------------------------------------------------------------------\n");

	//cpu 
	START_CPU
	h_result[0] = sumArrayByCpu(h_array);
	STOP_CPU

	printf("cpu \t\t\t\t%-12.0f\t%-21ld\t%f\n", h_result[0], end - begin, (end - begin) * 1.0 / time_cpu);

	//handle gpu memory
	HANDLE_ERROR(cudaMalloc((void**)&d_array, N * sizeof(REAL)));
	HANDLE_ERROR(cudaMalloc((void**)&d_result,1 * sizeof(REAL)));
	HANDLE_ERROR(cudaMemcpy(d_array, h_array, N * sizeof(REAL), cudaMemcpyHostToDevice));

	//gpu use Atomic with global memory
	HANDLE_ERROR(cudaMemset(d_result, 0, sizeof(REAL)));

	START_GPU
	sumArrayByGpuGlobal << < BOLCKS_PER_GRID, THREADS_PER_BLOCK >> > (d_result, d_array);
	STOP_GPU

	HANDLE_ERROR(cudaMemcpy(h_result, d_result, 1 * sizeof(REAL), cudaMemcpyDeviceToHost));
	printf("gpu + atomic + global \t\t%-12.0f\t%-21f\t%f\n", h_result[0], elapsed_time, elapsed_time / time_cpu);
	
	//gpu use Atomic with shared memory
	HANDLE_ERROR(cudaMemset(d_result, 0, sizeof(REAL)));

	START_GPU
	sumArrayByGpuShared << < BOLCKS_PER_GRID, THREADS_PER_BLOCK >> > (d_result, d_array);
	STOP_GPU

	HANDLE_ERROR(cudaMemcpy(h_result, d_result, 1 * sizeof(REAL), cudaMemcpyDeviceToHost));
	printf("gpu + atomic + shared \t\t%-12.0f\t%-21f\t%f\n", h_result[0], elapsed_time, elapsed_time / time_cpu);
	
	//gpu use Atomic with global memory
	HANDLE_ERROR(cudaMemset(d_result, 0, sizeof(REAL)));

	START_GPU
	sumArrayByGpuFewAtomic << < 22, 1024 >> > (d_result, d_array);
	STOP_GPU

	HANDLE_ERROR(cudaMemcpy(h_result, d_result, 1 * sizeof(REAL), cudaMemcpyDeviceToHost));
	printf("gpu + few atomic + global \t%-12.0f\t%-21f\t%f\n", h_result[0], elapsed_time, elapsed_time / time_cpu);

	HANDLE_ERROR(cudaFree(d_result));
	HANDLE_ERROR(cudaFree(d_array));
	
	free(h_array);
	free(h_result);

	return 0;
}