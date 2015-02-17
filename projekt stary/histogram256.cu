/*
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <helper_cuda.h>
#include "histogram_common.h"

inline __device__ void addByte(volatile uint *s_WarpHist, uint data, uint threadTag)
{
    uint count;

    do
    {
        count = s_WarpHist[data] & ( (1U << (UINT_BITS - LOG2_WARP_SIZE)) - 1U );
        count = threadTag | (count + 1);
        s_WarpHist[data] = count;
    }
    while (s_WarpHist[data] != count);
}

inline __device__ void addByteAtomic(uint *s_WarpHist, uint data, uint threadTag)
{
    atomicAdd(s_WarpHist + data, 1);
}

inline __device__ void addWord(uint *s_WarpHist, uint data, uint tag)
{
    addByte(s_WarpHist, (data >>  0) & 0xFFU, tag);
    addByte(s_WarpHist, (data >>  8) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 16) & 0xFFU, tag);
    addByte(s_WarpHist, (data >> 24) & 0xFFU, tag);
}

inline __device__ void addWordAtomic(uint *s_WarpHist, uint data, uint tag)
{
    addByteAtomic(s_WarpHist, (data >>  0) & 0xFFU, tag);
    addByteAtomic(s_WarpHist, (data >>  8) & 0xFFU, tag);
    addByteAtomic(s_WarpHist, (data >> 16) & 0xFFU, tag);
    addByteAtomic(s_WarpHist, (data >> 24) & 0xFFU, tag);
}

__global__ void histogram256Kernel(uint *d_PartialHistograms, uint *d_Data, uint dataCount)
{
    //Per-warp subhistogram storage
    __shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
    uint *s_WarpHist= s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

    //Clear shared memory storage for current threadblock before processing
#pragma unroll

    for (uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++)
    {
        s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
    }

    //Cycle through the entire data set, update subhistograms for each warp
    const uint tag = threadIdx.x << (UINT_BITS - LOG2_WARP_SIZE);

    __syncthreads();

    for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x))
    {
        uint data = d_Data[pos];
        addWord(s_WarpHist, data, tag);
    }

    //Merge per-warp histograms into per-block and write to global memory
    __syncthreads();

    for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE)
    {
        uint sum = 0;

        for (uint i = 0; i < WARP_COUNT; i++)
        {
            sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & ( (1U << (UINT_BITS - LOG2_WARP_SIZE)) - 1U );
        }

        d_PartialHistograms[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
    }
}

__global__ void shuffledUncoalescedAccesshistogram256Kernel(uint *d_PartialHistograms, uint *d_Data, uint dataCount)
{
	const int threadPos = 
        //[31 : 6] <== [31 : 6]
        ((threadIdx.x & (~63)) >> 0) |
        //[5  : 2] <== [3  : 0]
        ((threadIdx.x &    15) << 2) |
        //[1  : 0] <== [5  : 4]
        ((threadIdx.x &    48) >> 4);
	
    //Per-warp subhistogram storage
    __shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
    uint *s_WarpHist= s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

    //Clear shared memory storage for current threadblock before processing
#pragma unroll

    for (uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++)
    {
        s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
    }

    //Cycle through the entire data set, update subhistograms for each warp
    const uint tag = threadIdx.x << (UINT_BITS - LOG2_WARP_SIZE);

    __syncthreads();

    for (uint pos = UMAD(blockIdx.x, blockDim.x, threadPos); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x))
    {
        uint data = d_Data[pos];
        addWord(s_WarpHist, data, tag);
    }

    //Merge per-warp histograms into per-block and write to global memory
    __syncthreads();

    for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE)
    {
        uint sum = 0;

        for (uint i = 0; i < WARP_COUNT; i++)
        {
            sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & ( (1U << (UINT_BITS - LOG2_WARP_SIZE)) - 1U );
        }

        d_PartialHistograms[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
    }
}

// Atomic
__global__ void histogram256KernelAtomic(uint *d_PartialHistograms, uint *d_Data, uint dataCount)
{
    //Per-warp subhistogram storage
    __shared__ uint s_Hist[HISTOGRAM256_THREADBLOCK_MEMORY];
    uint *s_WarpHist= s_Hist + (threadIdx.x >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

    //Clear shared memory storage for current threadblock before processing
#pragma unroll

    for (uint i = 0; i < (HISTOGRAM256_THREADBLOCK_MEMORY / HISTOGRAM256_THREADBLOCK_SIZE); i++)
    {
        s_Hist[threadIdx.x + i * HISTOGRAM256_THREADBLOCK_SIZE] = 0;
    }

    //Cycle through the entire data set, update subhistograms for each warp
    const uint tag = threadIdx.x << (UINT_BITS - LOG2_WARP_SIZE);

    __syncthreads();

    for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x))
    {
        uint data = d_Data[pos];
        addWordAtomic(s_WarpHist, data, tag);
    }

    //Merge per-warp histograms into per-block and write to global memory
    __syncthreads();

    for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += HISTOGRAM256_THREADBLOCK_SIZE)
    {
        uint sum = 0;

        for (uint i = 0; i < WARP_COUNT; i++)
        {
            sum += s_Hist[bin + i * HISTOGRAM256_BIN_COUNT] & 0xFFFFFFFFU;
        }

        d_PartialHistograms[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Merge histogram256() output
// Run one threadblock per bin; each threadblock adds up the same bin counter
// from every partial histogram. Reads are uncoalesced, but mergeHistogram256
// takes only a fraction of total processing time
////////////////////////////////////////////////////////////////////////////////
#define MERGE_THREADBLOCK_SIZE 128

__global__ void mergeHistogram256Kernel(
    uint *d_Histogram,
    uint *d_PartialHistograms,
    uint histogramCount
)
{
    uint sum = 0;

    //for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE)//MERGE_THREADBLOCK_SIZE->HISTOGRAM256_BIN_COUNT ??
	for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE)//original
    {
        //sum += d_PartialHistograms[blockIdx.x + i * MERGE_THREADBLOCK_SIZE];
		sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM256_BIN_COUNT];//original
    }

    //__shared__ uint data[HISTOGRAM256_THREADBLOCK_SIZE];
	__shared__ uint data[MERGE_THREADBLOCK_SIZE];//original
    data[threadIdx.x] = sum;

    for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();

        //if (threadIdx.x < stride && threadIdx.x + stride < HISTOGRAM256_THREADBLOCK_SIZE)
		if (threadIdx.x < stride)//original
        {
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
    }
	
    if (threadIdx.x == 0)
    {
        d_Histogram[blockIdx.x] = data[0];
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host interface to GPU histogram
////////////////////////////////////////////////////////////////////////////////
//histogram256kernel() intermediate results buffer
static const uint PARTIAL_HISTOGRAM256_COUNT = 240;
static uint *d_PartialHistograms;

//Internal memory allocation
extern "C" void initHistogram256(void)
{
    checkCudaErrors(cudaMalloc((void **)&d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(uint)));
}

//Internal memory deallocation
extern "C" void closeHistogram256(void)
{
    checkCudaErrors(cudaFree(d_PartialHistograms));
}

extern "C" void histogram256(
    uint *d_Histogram,
    void *d_Data,
    uint byteCount
)
{
    assert(byteCount % sizeof(uint) == 0);
    histogram256Kernel<<<PARTIAL_HISTOGRAM256_COUNT, HISTOGRAM256_THREADBLOCK_SIZE>>>(d_PartialHistograms, (uint *)d_Data, byteCount / sizeof(uint));
    getLastCudaError("histogram256Kernel() execution failed\n");
	
	mergeHistogram256Kernel<<<HISTOGRAM256_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>(d_Histogram, d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT); //original gives error
    getLastCudaError("mergeHistogram256Kernel() execution failed\n");
}

#define TILE_DIM 32 //TODO try 16?
#define BLOCK_ROWS 8 //TODO try other

inline __device__ int shuffleBits(int input)
{
	return (input & (~0x111)) | ((input & 0x011) << 2) | ((input & 0x100) >> 2);
}

//original
__global__ void shuffleCoalesced(uchar *odata, const uchar *idata, const uint width)
{
	__shared__ uchar tile[TILE_DIM][TILE_DIM+1];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

	__syncthreads();

	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
		odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

//no bank conflicts in filling tile (somewhat unstable)
//__global__ void shuffleCoalesced(uchar *odata, const uchar *idata, const uint width)
//{
//	__shared__ uchar tile[TILE_DIM][(TILE_DIM+1)<<2];
//
//	int x = blockIdx.x * TILE_DIM + threadIdx.x;
//	int y = blockIdx.y * TILE_DIM + threadIdx.y;
//	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
//		tile[threadIdx.y+j][threadIdx.x<<2] = idata[(y+j)*width + x];
//
//	__syncthreads();
//
//	for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
//		odata[(y+j)*width + x] = tile[threadIdx.x][(threadIdx.y + j)<<2];
//}

extern "C" void shuffledHistogram(
	uint *d_Histogram,
	void *d_Shuffled_Data,
    void *d_Data,
    uint byteCount
)
{
	assert(byteCount % 2048 == 0);
	const int nx = 2048;
	const int ny = byteCount/nx;
	dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
	shuffleCoalesced <<<dimGrid, dimBlock>>> ((uchar *) d_Shuffled_Data, (uchar *) d_Data, dimGrid.x * TILE_DIM);
	getLastCudaError("shuffleCoalesced() execution failed\n");
	//TODO remove synchronization (used only to pinpoint error)
	checkCudaErrors(cudaDeviceSynchronize());
	
	histogram256(d_Histogram, (void *) d_Shuffled_Data, byteCount);
}

extern "C" void extension2Histogram(
	uint *d_Histogram,
    void *d_Data,
    uint byteCount
)
{
	assert(byteCount % 64 == 0);
    shuffledUncoalescedAccesshistogram256Kernel<<<PARTIAL_HISTOGRAM256_COUNT, HISTOGRAM256_THREADBLOCK_SIZE>>>(d_PartialHistograms, (uint *)d_Data, byteCount / sizeof(uint));
    getLastCudaError("histogram256Kernel() execution failed\n");
	
	mergeHistogram256Kernel<<<HISTOGRAM256_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>(d_Histogram, d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT); //original gives error
    getLastCudaError("mergeHistogram256Kernel() execution failed\n");
}

extern "C" void shuffleData(
	void *d_Shuffled_Data,
    void *d_Data,
    uint byteCount
)
{
	assert(byteCount % 2048 == 0);
	const int nx = 2048;
	const int ny = byteCount/nx;
	dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
	shuffleCoalesced <<<dimGrid, dimBlock>>> ((uchar *) d_Shuffled_Data, (uchar *) d_Data, dimGrid.x * TILE_DIM);
	getLastCudaError("shuffleCoalesced() execution failed\n");
}

#ifdef CUDA_NO_SM12_ATOMIC_INTRINSICS
#error Compilation target does not support shared-memory atomics
#endif

extern "C" void extension3Histogram(
    uint *d_Histogram,
    void *d_Data,
    uint byteCount
)
{
    assert(byteCount % sizeof(uint) == 0);
    histogram256KernelAtomic<<<PARTIAL_HISTOGRAM256_COUNT, HISTOGRAM256_THREADBLOCK_SIZE>>>(d_PartialHistograms, (uint *)d_Data, byteCount / sizeof(uint));
    getLastCudaError("histogram256Kernel() execution failed\n");
	
	mergeHistogram256Kernel<<<HISTOGRAM256_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>(d_Histogram, d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT); //original gives error
    getLastCudaError("mergeHistogram256Kernel() execution failed\n");
}

#define SHARED_MEMORY_BANK_COUNT 32

//run with block of 32 threads, sums to global histogram
__global__ void threadHistogram256Kernel(uint *d_Histogram, uint *d_Data, uint dataCount)
{
	//Per-thread subhistogram storage
	//__shared__ uint threadHist[HISTOGRAM256_BIN_COUNT << 5];
	__shared__ uint threadHist[HISTOGRAM256_BIN_COUNT][SHARED_MEMORY_BANK_COUNT];
	
	uint bank = threadIdx.x % SHARED_MEMORY_BANK_COUNT;
	
	//clear subhistogram for current threadIdx
	for(int bin = 0; bin < HISTOGRAM256_BIN_COUNT; bin++)
		//threadHist[(bin << 5) | bank] = 0;
		threadHist[bin][bank] = 0;
	
	__syncthreads();
	
	//Cycle through the entire data set, update subhistograms for each thread
	for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x))
    {
		uint data4 = d_Data[pos];
		//threadHist[(((data4 >>  0) & 0xFFU) << 5) | bank]++;
		//threadHist[(((data4 >>  8) & 0xFFU) << 5) | bank]++;
		//threadHist[(((data4 >> 16) & 0xFFU) << 5) | bank]++;
		//threadHist[(((data4 >> 24) & 0xFFU) << 5) | bank]++;
		threadHist[(data4 >>  0) & 0xFFU][bank]++;
		threadHist[(data4 >>  8) & 0xFFU][bank]++;
		threadHist[(data4 >> 16) & 0xFFU][bank]++;
		threadHist[(data4 >> 24) & 0xFFU][bank]++;
    }
	
	__syncthreads();
	
	for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += SHARED_MEMORY_BANK_COUNT)
	{
		register uint sum = 0;
		for (uint bnk = 0; bnk < SHARED_MEMORY_BANK_COUNT; bnk++)
			//sum += threadHist[(bin << 5) | ((threadIdx.x + bnk) & 31)];
			sum += threadHist[bin][(threadIdx.x + bnk) & 31];
		d_Histogram[bin] += sum;
	}
}

//sums to partial
__global__ void threadPartialHistogram256Kernel(uint *d_PartialHistograms, uint *d_Data, uint dataCount)
{
	//Per-thread subhistogram storage
	//__shared__ uint threadHist[HISTOGRAM256_BIN_COUNT << 5];
	__shared__ uint threadHist[HISTOGRAM256_BIN_COUNT][SHARED_MEMORY_BANK_COUNT];
	
	uint bank = threadIdx.x % SHARED_MEMORY_BANK_COUNT;
	
	//clear subhistogram for current threadIdx
	for(int bin = 0; bin < HISTOGRAM256_BIN_COUNT; bin++)
		//threadHist[(bin << 5) | bank] = 0;
		threadHist[bin][bank] = 0;
	
	__syncthreads();
	
	//Cycle through the entire data set, update subhistograms for each thread
	for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x))
    {
		uint data4 = d_Data[pos];
		//threadHist[(((data4 >>  0) & 0xFFU) << 5) | bank]++;
		//threadHist[(((data4 >>  8) & 0xFFU) << 5) | bank]++;
		//threadHist[(((data4 >> 16) & 0xFFU) << 5) | bank]++;
		//threadHist[(((data4 >> 24) & 0xFFU) << 5) | bank]++;
		threadHist[(data4 >>  0) & 0xFFU][bank]++;
		threadHist[(data4 >>  8) & 0xFFU][bank]++;
		threadHist[(data4 >> 16) & 0xFFU][bank]++;
		threadHist[(data4 >> 24) & 0xFFU][bank]++;
    }
	
    //Merge per-warp histograms into per-block and write to global memory
    __syncthreads();
	
	for (uint bin = threadIdx.x; bin < HISTOGRAM256_BIN_COUNT; bin += SHARED_MEMORY_BANK_COUNT)
	{
		register uint sum = 0;
		for (uint bnk = 0; bnk < SHARED_MEMORY_BANK_COUNT; bnk++)
			//sum += threadHist[(bin << 5) | ((threadIdx.x + bnk) & 31)];
			sum += threadHist[bin][(threadIdx.x + bnk) & 31];
		d_PartialHistograms[blockIdx.x * HISTOGRAM256_BIN_COUNT + bin] = sum;
	}
}

__global__ void clearHistogram(uint *d_Histogram, uint size)
{
	//clear histogram
	for (uint bin = UMAD(blockIdx.x, blockDim.x, threadIdx.x); bin < size; bin += UMUL(blockDim.x, gridDim.x))
		d_Histogram[bin] = 0;
}

extern "C" void threadHistogram(
    uint *d_Histogram,
    void *d_Data,
    uint byteCount
)
{
	assert(byteCount % SHARED_MEMORY_BANK_COUNT == 0);
	dim3 dimGrid(5, 1, 1);
	dim3 dimBlock(SHARED_MEMORY_BANK_COUNT, 1, 1);
	clearHistogram<<<1,HISTOGRAM256_BIN_COUNT>>>(d_Histogram,HISTOGRAM256_BIN_COUNT);
	getLastCudaError("clearHistogram() execution failed\n");
	threadHistogram256Kernel<<<dimGrid, dimBlock>>>(d_Histogram, (uint *) d_Data, byteCount / sizeof(uint));
	getLastCudaError("threadHistogram256Kernel() execution failed\n");
}

extern "C" void threadPartialHistogram(
    uint *d_Histogram,
    void *d_Data,
    uint byteCount
)
{
	assert(byteCount % SHARED_MEMORY_BANK_COUNT == 0);
	dim3 dimGrid(192, 1, 1);
	dim3 dimBlock(SHARED_MEMORY_BANK_COUNT, 1, 1);
	clearHistogram<<<1,HISTOGRAM256_BIN_COUNT>>>(d_Histogram, HISTOGRAM256_BIN_COUNT);
	getLastCudaError("clearHistogram() execution failed\n");
	clearHistogram<<<PARTIAL_HISTOGRAM256_COUNT,HISTOGRAM256_BIN_COUNT>>>(d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT*HISTOGRAM256_BIN_COUNT);
	getLastCudaError("clearHistogram() execution failed\n");
	threadPartialHistogram256Kernel<<<dimGrid, dimBlock>>>(d_PartialHistograms, (uint *) d_Data, byteCount / sizeof(uint));
	getLastCudaError("threadHistogram256Kernel() execution failed\n");
	mergeHistogram256Kernel<<<HISTOGRAM256_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>(d_Histogram, d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT);
	getLastCudaError("mergeHistogram256Kernel() execution failed\n");
}



