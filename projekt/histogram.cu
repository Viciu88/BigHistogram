#include <helper_cuda.h>

#define UMUL(a, b) ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;


#define SHARED_MEMORY_SIZE 49152
#define MERGE_THREADBLOCK_SIZE 128

/*
 *	Function that maps value to bin in range 0 inclusive to binCOunt exclusive
 */
inline __device__ uint binOfValue(uint value, uint binCount)
{
	//TODO get some sensible function to assign bin
	return value % binCount;
}

inline __device__ void addAtomic(uint *s_WarpHist, uint bin)
{
    atomicAdd(s_WarpHist + bin, 1);
}

inline __device__ uint untag(uint taggedValue)
{
	return taggedValue & ((1U << 27) - 1U);
}

inline __device__ uint tag(uint value, uint tag)
{
	return tag | value;
}

inline __device__ void addTagged(volatile uint *s_WarpHist, uint bin, uint threadTag)
{
	uint tmp;
    do
    {
		tmp = untag(s_WarpHist[bin]);
		tmp = tag(tmp + 1, threadTag);
        s_WarpHist[bin] = tmp;//update shared memory with new value and tag
    }
    while (s_WarpHist[bin] != tmp);//until race won
}

__global__ void clearHistogram(uint *d_Histogram, uint binCount)
{
	//clear histogram
	for (uint bin = UMAD(blockIdx.x, blockDim.x, threadIdx.x); bin < binCount; bin += UMUL(blockDim.x, gridDim.x))
		d_Histogram[bin] = 0;
}

//1 byte per bin kernel
__global__ void byteHistogramKernel(uint *d_PartialHistograms, uint *d_Data, uint dataCount, uint binCount)
{
	//TODO move constants out of kernel
	uint tid = UMAD(blockIdx.x, blockDim.x, threadIdx.x);
	uint threadCount = UMUL(blockDim.x, gridDim.x);
	uint binsPerThread = binCount / blockDim.x;
	
	//TODO try to limit bank conflicts
	extern __shared__ uchar s_byteHistogram[];
	//__shared__ uchar s_byteHistogram[SHARED_MEMORY_SIZE];

	//clear shared memory histogram bins assigned to this thread
	#pragma unroll
	for (uint bin = binsPerThread * threadIdx.x; bin < binsPerThread * (threadIdx.x + 1) && bin < binCount; bin++)
		s_byteHistogram[bin] = 0;
	__syncthreads();
	
	for (uint data = tid; data < dataCount; data += threadCount)//approximate
	//for (uint data = 0; data < dataCount; data++)//with do over
	{
		uint bin = binOfValue(d_Data[data], binCount);
		if(bin >= binsPerThread * threadIdx.x && bin < binsPerThread * (threadIdx.x + 1))
		{
			//update bin (no need for synchronization, only this thread can modify this bin)
			s_byteHistogram[bin]++;
			//if overflow copy to global memory
			if(s_byteHistogram[bin] == 255)
			{
				d_PartialHistograms[blockIdx.x * binCount + bin] += s_byteHistogram[bin];
				s_byteHistogram[bin] = 0;
			}
		}
		else
		{
			//disregard data has to be processed by other thread
		}
	}
	
	//copy final histogram bins assigned to this thread to global
	#pragma unroll
	for (uint bin = binsPerThread * threadIdx.x; bin < binsPerThread * (threadIdx.x + 1) && bin < binCount; bin++)
		d_PartialHistograms[blockIdx.x * binCount + bin] += s_byteHistogram[bin];
}

__global__ void shortHistogramKernel(uint *d_PartialHistograms, uint *d_Data, uint dataCount, uint binCount)
{
	//TODO move constants out of kernel
	uint tid = UMAD(blockIdx.x, blockDim.x, threadIdx.x);
	uint threadCount = UMUL(blockDim.x, gridDim.x);
	uint binsPerThread = binCount / blockDim.x;
	
	//TODO try to limit bank conflicts
	extern __shared__ ushort s_shortHistogram[];
	//__shared__ ushort s_shortHistogram[SHARED_MEMORY_SIZE/2];

	//clear shared memory histogram bins assigned to this thread
	#pragma unroll
	for (uint bin = binsPerThread * threadIdx.x; bin < binsPerThread * (threadIdx.x + 1) && bin < binCount; bin++)
		s_shortHistogram[bin] = 0;
	__syncthreads();
	
	for (uint data = tid; data < dataCount; data += threadCount)//approximate
	//for (uint data = 0; data < dataCount; data++)//with do over
	{
		uint bin = binOfValue(d_Data[data], binCount);
		if(bin >= binsPerThread * threadIdx.x && bin < binsPerThread * (threadIdx.x + 1))
		{
			//update bin (no need for synchronization, only this thread can modify this bin)
			s_shortHistogram[bin]++;
			//if overflow copy to global memory
			if(s_shortHistogram[bin] == 65535)
			{
				d_PartialHistograms[blockIdx.x * binCount + bin] += s_shortHistogram[bin];
				s_shortHistogram[bin] = 0;
			}
		}
		else
		{
			//disregard data has to be processed by other thread
		}
	}
	
	//copy final histogram bins assigned to this thread to global
	#pragma unroll
	for (uint bin = binsPerThread * threadIdx.x; bin < binsPerThread * (threadIdx.x + 1) && bin < binCount; bin++)
		d_PartialHistograms[blockIdx.x * binCount + bin] += s_shortHistogram[bin];
}

__global__ void intHistogramKernel(uint *d_PartialHistograms, uint *d_Data, uint dataCount, uint binCount)
{
	//TODO move constants out of kernel
	uint tid = UMAD(blockIdx.x, blockDim.x, threadIdx.x);
	uint threadCount = UMUL(blockDim.x, gridDim.x);
	uint binsPerThread = binCount / blockDim.x;
	
	//TODO try to limit bank conflicts
	extern __shared__ uint s_Histogram[];
	//__shared__ uint s_Histogram[SHARED_MEMORY_SIZE/4];

	//clear shared memory histogram bins assigned to this thread
	#pragma unroll
	for (uint bin = binsPerThread * threadIdx.x; bin < binsPerThread * (threadIdx.x + 1) && bin < binCount; bin++)
		s_Histogram[bin] = 0;
	__syncthreads();
	
	for (uint data = tid; data < dataCount; data += threadCount)//approximate
	//for (uint data = 0; data < dataCount; data++)//with do over
	{
		uint bin = binOfValue(d_Data[data], binCount);
		if(bin >= binsPerThread * threadIdx.x && bin < binsPerThread * (threadIdx.x + 1))
		{
			//update bin (no need for synchronization, only this thread can modify this bin)
			s_Histogram[bin]++;
		}
		else
		{
			//disregard data has to be processed by other thread
		}
	}
	
	//copy final histogram bins assigned to this thread to global
	#pragma unroll
	for (uint bin = binsPerThread * threadIdx.x; bin < binsPerThread * (threadIdx.x + 1) && bin < binCount; bin++)
		d_PartialHistograms[blockIdx.x * binCount + bin] += s_Histogram[bin];
}

__global__ void mergePartialHistogramsKernel(uint *d_Histogram, uint *d_PartialHistograms, uint histogramCount,	uint binCount)
{
	for (uint bin = blockIdx.x; bin < binCount; bin += gridDim.x)
	{
		uint sum = 0;
		for (uint histogramIndex = threadIdx.x; histogramIndex < histogramCount; histogramIndex += MERGE_THREADBLOCK_SIZE)
		{
			sum += d_PartialHistograms[bin + histogramIndex * binCount];
		}
	
		__shared__ uint data[MERGE_THREADBLOCK_SIZE];
		data[threadIdx.x] = sum;
	
		for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1)
		{
			__syncthreads();
	
			if (threadIdx.x < stride)
			{
				data[threadIdx.x] += data[threadIdx.x + stride];
			}
		}
		
		if (threadIdx.x == 0)
		{
			d_Histogram[bin] = data[0];
		}
	}
}

__global__ void baseHistogramKernel(uint *d_PartialHistograms, uint *d_Data, uint dataCount, uint binCount)
{
	//TODO move constants out of kernel
	uint tid = UMAD(blockIdx.x, blockDim.x, threadIdx.x);
	uint threadCount = UMUL(blockDim.x, gridDim.x);
	
	//TODO fix Tagged arithmetic
	//assumed warpCount * binCount int cells in shared memory
	uint warpCount = 1;
	uint sharedMemorySizeUsed = warpCount * binCount;
	
	extern __shared__ uint s_Histogram[];
	
	uint warpSize = 32;
	uint warpIndex = threadIdx.x / warpSize; //(threadIdx.x >> LOG2_WARP_SIZE); 
	uint warpHistogramIndex = warpIndex % warpCount;
	uint *s_WarpHist = s_Histogram + warpHistogramIndex * binCount;
	
	//clear shared memory for threadblock //histogram bins assigned to this thread
	#pragma unroll
	for (uint bin = threadIdx.x; bin < sharedMemorySizeUsed; bin += blockDim.x)
		s_Histogram[bin] = 0;
	
	__syncthreads();
	
	const uint tag = threadIdx.x << 27;
	for (uint pos = tid; pos < dataCount; pos += threadCount)
	{
		uint data = d_Data[pos];
		uint bin = binOfValue(data, binCount);
		
		//atomic add 1 //s_WarpHist[bin]++;
		addTagged(s_WarpHist, bin, tag); //with tag
	}
	__syncthreads();

    for (uint bin = threadIdx.x; bin < binCount; bin += blockDim.x)
    {
        uint sum = 0;
        for (uint i = 0; i < warpCount; i++)
        {
			sum += untag(s_Histogram[bin + i * binCount]);
            //sum += s_Histogram[bin + i * binCount] & ( (1U << 27) - 1U ); // with tag removed
        }
        d_PartialHistograms[blockIdx.x * binCount + bin] = sum;
    }
}

__global__ void baseHistogramKernelAtomic (uint *d_PartialHistograms, uint *d_Data, uint dataCount, uint binCount)
{
	//TODO move constants out of kernel
	uint tid = UMAD(blockIdx.x, blockDim.x, threadIdx.x);
	uint threadCount = UMUL(blockDim.x, gridDim.x);
	
	//assumed warpCount * binCount int cells in shared memory
	uint warpCount = 1;
	uint sharedMemorySizeUsed = warpCount * binCount;
	
	extern __shared__ uint s_Histogram[];
	
	uint warpSize = 32;
	uint warpIndex = threadIdx.x / warpSize; //(threadIdx.x >> LOG2_WARP_SIZE); 
	uint warpHistogramIndex = warpIndex % warpCount;
	uint *s_WarpHist = s_Histogram + warpHistogramIndex * binCount;
	
	//clear shared memory for threadblock //histogram bins assigned to this thread
	#pragma unroll
	for (uint bin = threadIdx.x; bin < sharedMemorySizeUsed; bin += blockDim.x)
		s_Histogram[bin] = 0;
	
	__syncthreads();
	
	for (uint pos = tid; pos < dataCount; pos += threadCount)
	{
		uint data = d_Data[pos];
		uint bin = binOfValue(data, binCount);
		
		//atomic add 1 //s_WarpHist[bin]++;
		addAtomic(s_WarpHist, bin); //without tag
	}
	__syncthreads();

    for (uint bin = threadIdx.x; bin < binCount; bin += blockDim.x)
    {
        uint sum = 0;
        for (uint i = 0; i < warpCount; i++)
        {
			sum += s_Histogram[bin + i * binCount];//without tag
        }
        d_PartialHistograms[blockIdx.x * binCount + bin] = sum;
    }
}

static uint *d_PartialHistograms;

extern "C" void initPartialHistograms(uint partialHistogramCount, uint binCount)
{
    checkCudaErrors(cudaMalloc((void **)&d_PartialHistograms, partialHistogramCount * binCount * sizeof(uint)));
}

//Internal memory deallocation
extern "C" void closePartialHistograms(void)
{
    checkCudaErrors(cudaFree(d_PartialHistograms));
}

extern "C" void clearHistogramsAndPartialHistograms(uint *d_Histogram, uint partialHistogramCount, uint binCount)
{
	clearHistogram<<<partialHistogramCount, 512>>>(d_Histogram, binCount);
	getLastCudaError("clearHistogram() execution failed\n");
	clearHistogram<<<partialHistogramCount, 512>>>(d_PartialHistograms, partialHistogramCount * binCount);
	getLastCudaError("clearHistogram() execution failed\n");
}

extern "C" void approxHistogramGPU(uint *d_Histogram, void *d_Data, uint byteCount, uint binCount, cudaDeviceProp deviceProp)
{
	uint partialHistogramCount = 128;
	initPartialHistograms(partialHistogramCount, binCount);
	
	clearHistogramsAndPartialHistograms(d_Histogram, partialHistogramCount, binCount);
	
	//dynamically get bytes per bin depending on hardware
	uint bytesPerBin = deviceProp.sharedMemPerBlock / binCount;
	
	if(bytesPerBin == 0)
	{
		// Too many bins. Cannot be processed on given hardware
		printf("... execution failed too many bins\n");
	}
	else if (bytesPerBin == 1)
	{
		printf("... using byteHistogramKernel\n");
		//use kernel with 1 byte per bin
		byteHistogramKernel<<<partialHistogramCount, 256, binCount * sizeof(uchar) >>>(d_PartialHistograms, (uint *) d_Data, byteCount / sizeof(uint), binCount);
		cudaDeviceSynchronize();
		getLastCudaError("byteHistogramKernel() execution failed\n");

		mergePartialHistogramsKernel<<<256, MERGE_THREADBLOCK_SIZE>>>(d_Histogram, d_PartialHistograms, partialHistogramCount, binCount);
		cudaDeviceSynchronize();
		getLastCudaError("mergePartialHistogramsKernel() execution failed\n");
	}
	else if (bytesPerBin == 2 || bytesPerBin == 3)
	{
		printf("... using shortHistogramKernel\n");
		//use kernel with 2 byte per bin
		shortHistogramKernel<<<partialHistogramCount, 256, binCount * sizeof(ushort) >>>(d_PartialHistograms, (uint *) d_Data, byteCount / sizeof(uint), binCount);
		getLastCudaError("shortHistogramKernel() execution failed\n");

		mergePartialHistogramsKernel<<<256, MERGE_THREADBLOCK_SIZE>>>(d_Histogram, d_PartialHistograms, partialHistogramCount, binCount);
		getLastCudaError("mergePartialHistogramsKernel() execution failed\n");
	}
	else if (bytesPerBin > 3 )
	{
		printf("... using intHistogramKernel\n");
		//use kernel with 4 byte per bin
		intHistogramKernel<<<partialHistogramCount, 256, binCount * sizeof(uint) >>>(d_PartialHistograms, (uint *) d_Data, byteCount / sizeof(uint), binCount);
		getLastCudaError("intHistogramKernel() execution failed\n");

		mergePartialHistogramsKernel<<<256, MERGE_THREADBLOCK_SIZE>>>(d_Histogram, d_PartialHistograms, partialHistogramCount, binCount);
		getLastCudaError("mergePartialHistogramsKernel() execution failed\n");
	}
	
	closePartialHistograms();
}

extern "C" void baseHistogramGPU(uint *d_Histogram, void *d_Data, uint byteCount, uint binCount, cudaDeviceProp deviceProp)
{
	uint partialHistogramCount = 128;
	initPartialHistograms(partialHistogramCount, binCount);
	
	clearHistogramsAndPartialHistograms(d_Histogram, partialHistogramCount, binCount);
	
	if(deviceProp.sharedMemPerBlock < binCount * sizeof(uint))
	{
		// Too many bins. Cannot be processed on given hardware
		printf("... execution failed too many bins\n");
		return;
	}
	//kernels
	printf("... using baseHistogramKernel\n");
	//use kernel with 4 byte per bin
	//TODO add more warphistograms if possible if(shared memory /(binCount*4) > 1)
	baseHistogramKernel<<<partialHistogramCount, 256, binCount * sizeof(uint) >>>(d_PartialHistograms, (uint *) d_Data, byteCount / sizeof(uint), binCount);
	getLastCudaError("baseHistogramKernel() execution failed\n");

	mergePartialHistogramsKernel<<<256, MERGE_THREADBLOCK_SIZE>>>(d_Histogram, d_PartialHistograms, partialHistogramCount, binCount);
	getLastCudaError("mergePartialHistogramsKernel() execution failed\n");
	
	closePartialHistograms();
}

extern "C" void baseHistogramAtomicGPU(uint *d_Histogram, void *d_Data, uint byteCount, uint binCount, cudaDeviceProp deviceProp)
{
	uint partialHistogramCount = 128;
	initPartialHistograms(partialHistogramCount, binCount);
	
	clearHistogramsAndPartialHistograms(d_Histogram, partialHistogramCount, binCount);
	
	if(deviceProp.sharedMemPerBlock < binCount * sizeof(uint))
	{
		// Too many bins. Cannot be processed on given hardware
		printf("... execution failed too many bins\n");
		return;
	}
	//kernels
	printf("... using baseHistogramKernelAtomic\n");
	//use kernel with 4 byte per bin
	//TODO add more warphistograms if possible if(shared memory /(binCount*4) >1)
	baseHistogramKernelAtomic<<<partialHistogramCount, 256, binCount * sizeof(uint) >>>(d_PartialHistograms, (uint *) d_Data, byteCount / sizeof(uint), binCount);
	getLastCudaError("baseHistogramKernelAtomic() execution failed\n");

	mergePartialHistogramsKernel<<<256, MERGE_THREADBLOCK_SIZE>>>(d_Histogram, d_PartialHistograms, partialHistogramCount, binCount);
	getLastCudaError("mergePartialHistogramsKernel() execution failed\n");
	
	closePartialHistograms();
}