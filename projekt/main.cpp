// CUDA Runtime
#include <cuda_runtime.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA SDK samples

#include "histogram.h"

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

#define DEBUG_OUTPUT 1

inline uint bin(uint data, uint binCount)
{
	return data % binCount;
}

void histogramCPU(uint *h_Histogram, void *h_Data, uint byteCount, uint binCount)
{
#if(DEBUG_OUTPUT)
	printf("HistogramCPU()...\n");
#endif
    for (uint i = 0; i < binCount; i++)
        h_Histogram[i] = 0;

    assert(sizeof(uint) == 4 && (byteCount % 4) == 0);

    for (uint i = 0; i < (byteCount / 4); i++)
    {
        uint data = ((uint *)h_Data)[i];
		h_Histogram[bin(data, binCount)]++;
    }
}

void generateRandomData(uchar *h_Data, uint byteCount)
{
#if(DEBUG_OUTPUT)
	printf("...generating random input data\n");
#endif
	srand(2009);
	for (uint i = 0; i < byteCount; i++)
		h_Data[i] = rand() % 256;
}

cudaDeviceProp checkCudaDevice(uint argc, const char ** argv)
{
	cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
	//Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int dev = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
#if(DEBUG_OUTPUT)
    printf("CUDA device [%s] has %d Multi-Processors, Compute %d.%d\n",deviceProp.name, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
#endif

    int version = deviceProp.major * 0x10 + deviceProp.minor;

    if (version < 0x11)
    {
        printf("There is no device supporting a minimum of CUDA compute capability 1.1 for this SDK sample\n");
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
	return deviceProp;
}

uint getBinCount(uint argc, const char ** argv)
{
	for(int i = 1; i < argc; i++)
		if(strcmp(argv[i], "-bin") == 0 && argc > i + 1)
			return atoi(argv[i + 1]);
	return 48000;
}

void compareHistogram(uint *h_HistogramCPU, uint *h_HistogramGPU, uint binCount)
{
	int PassFailFlag = 1;
#if(DEBUG_OUTPUT)
    printf("Comparing the results...\n");
#endif
	uint errorCount=0;
    for (uint i = 0; i < binCount; i++)
        if (h_HistogramGPU[i] != h_HistogramCPU[i])
        {
			if(errorCount++ < 10)
				printf("error: bin[%d]=%d expected:%d\n", i, h_HistogramGPU[i], h_HistogramCPU[i]);
            PassFailFlag = 0;
        }
#if(DEBUG_OUTPUT)
	printf(PassFailFlag ? "...histograms match\n\n" : "...histograms do not match!\n\n");
#else
	if(!PassFailFlag)//print if fails
		printf("Histograms do not match!\n\n");
#endif
}

int main(int argc, char **argv)
{
	//initialize device
	cudaDeviceProp deviceProp = checkCudaDevice(argc, (const char **)argv);

	uchar *h_Data;
    uint byteCount = 64 * 1048576;
	uint numRuns = 16;
	uint *h_HistogramCPU;
	uint *h_HistogramGPU;
    uchar *d_Data;
    uint *d_Histogram;
	uint binCount = getBinCount(argc, (const char **)argv);
#if(DEBUG_OUTPUT)
    printf("Histogram with %d bins\n", binCount);
#endif

	//allocate host data
#if(DEBUG_OUTPUT)
    printf("Initializing data...\n");
    printf("...allocating CPU memory\n");
#endif
    h_Data         = (uchar *)malloc(byteCount);
    h_HistogramCPU = (uint *)malloc(binCount * sizeof(uint));
    h_HistogramGPU = (uint *)malloc(binCount * sizeof(uint));

	//allocate device data
	cudaDeviceReset();
#if(DEBUG_OUTPUT)
    printf("...allocating GPU memory\n");
#endif
    checkCudaErrors(cudaMalloc((void **)&d_Data, byteCount));
    checkCudaErrors(cudaMalloc((void **)&d_Histogram, binCount * sizeof(uint)));

	//initialize host data
#if(DEBUG_OUTPUT)
    printf("Loading data...\n");
#endif
	generateRandomData(h_Data, byteCount);//TODO other methods

	histogramCPU(h_HistogramCPU, h_Data, byteCount, binCount);

	//initialize device data
#if(DEBUG_OUTPUT)
    printf("Copying data to device...\n");
#endif
	checkCudaErrors(cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));

	//determine execution configuration
	//execute kernel
#if(DEBUG_OUTPUT)
    printf("HistogramGPU()...\n");
#endif
	uint gridSize = 128;
	uint blockSize = 256;
	baseHistogramGPU(d_Histogram, d_Data, byteCount, binCount, gridSize, blockSize, deviceProp);

	//transfer resulting data to host
#if(DEBUG_OUTPUT)
    printf("Copying histogram to host...\n");
#endif
	checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, binCount * sizeof(uint), cudaMemcpyDeviceToHost));

	//check validity of results
	compareHistogram(h_HistogramCPU, h_HistogramGPU, binCount);

	//memory deallocation and cleanup
#if(DEBUG_OUTPUT)
    printf("Shutting down...\n");
#endif
    checkCudaErrors(cudaFree(d_Histogram));
    checkCudaErrors(cudaFree(d_Data));
    free(h_HistogramGPU);
    free(h_HistogramCPU);
    free(h_Data);

    cudaDeviceReset();
	exit(EXIT_SUCCESS);
}
