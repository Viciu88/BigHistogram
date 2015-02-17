// CUDA Runtime
#include <cuda_runtime.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA SDK samples

// project include
#include "histogram_common.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>

#define DEBUG_OUTPUT 0

const int numRuns = 16;

void loadDataFromImage(uchar *h_Data, uint byteCount, std::string filename)
{
#if(DEBUG_OUTPUT)
	std::cout << "Loading data from image: " << filename << std::endl;
#endif
	cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if(! image.data )                             
	{
		printf("Could not open or find the image\n");
		exit(EXIT_FAILURE);
	}
	
	uint k = 0;
	for(int j=0;j<image.rows;j++) 
	{
		for (int i=0;i<image.cols;i++)
		{
			if(k>byteCount)
				return;
			h_Data[k] =  (int)image.at<uchar>(i,j);
			k++;
		}
	}
	//filling to the end of data with copy of input
	uint imageBytes = k;
	for(; k<byteCount; k++)
		h_Data[k] = h_Data[k - imageBytes];
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

void generateDegenerateData(uchar *h_Data, uint byteCount)
{
#if(DEBUG_OUTPUT)
	printf("...generating degenerate input data\n");
#endif
	srand(2009);
	uint rnd = rand() % 256;
	for (uint i = 0; i < byteCount; i++)
		h_Data[i] = rnd;
}

int podlozhnyukHistogram(uchar *d_Data, uint  *d_Histogram, uint byteCount, uchar *h_Data, uint  *h_HistogramCPU, uint *h_HistogramGPU)
{
	int PassFailFlag = 1;
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
#if(DEBUG_OUTPUT)
    printf("Initializing 256-bin histogram...\n");
#endif
    initHistogram256();
#if(DEBUG_OUTPUT)
    printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);
#endif
    for (int iter = -1; iter < numRuns; iter++)
    {
        //iter == -1 -- warmup iteration
        if (iter == 0)
        {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }
        histogram256(d_Histogram, d_Data, byteCount);
    }
    
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
#if(DEBUG_OUTPUT)
    printf("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
    printf("histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM256_THREADBLOCK_SIZE);
#else
	printf("Podlozhnyuk histogram: %.4f MB/s \t %.5f s\n", (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs );
#endif
    
#if(DEBUG_OUTPUT)
    printf("\nValidating GPU results...\n");
    printf(" ...reading back GPU results\n");
#endif
    checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
#if(DEBUG_OUTPUT)
    printf(" ...comparing the results\n");
#endif
	uint errorCount=0;
    for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
        if (h_HistogramGPU[i] != h_HistogramCPU[i])
        {
			if(errorCount++<10)
				printf("error: bin[%d]=%d expected:%d\n",i,h_HistogramGPU[i],h_HistogramCPU[i]);
            PassFailFlag = 0;
        }
#if(DEBUG_OUTPUT)
	printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");
#else
	if(!PassFailFlag)//print if fails
		printf(" ***256-bin histograms do not match!!!***\n\n");
#endif
    
#if(DEBUG_OUTPUT)
    printf("Shutting down 256-bin histogram...\n\n\n");
#endif
    closeHistogram256();
	return PassFailFlag;
}

int extension1Histogram(uchar *d_Data, uint  *d_Histogram, uint byteCount, uchar *h_Data, uint  *h_HistogramCPU, uint *h_HistogramGPU)
{
	int PassFailFlag = 1;
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	//shuffled data allocation
	uchar *d_Shuffled_Data;
	checkCudaErrors(cudaMalloc((void **)&d_Shuffled_Data, byteCount));
	
#if(DEBUG_OUTPUT)
    printf("Initializing 256-bin histogram...\n");
#endif
    initHistogram256();
#if(DEBUG_OUTPUT)
    printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);
#endif
    for (int iter = -1; iter < numRuns; iter++)
    {
        //iter == -1 -- warmup iteration
        if (iter == 0)
        {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }
        //shuffleData(d_Shuffled_Data, d_Data, byteCount);
		//histogram256(d_Histogram, d_Shuffled_Data, byteCount);
		shuffledHistogram(d_Histogram, d_Shuffled_Data, d_Data, byteCount);
    }
    
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
#if(DEBUG_OUTPUT)
    printf("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
    printf("histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM256_THREADBLOCK_SIZE);
#else
	printf("Extension 1 histogram: %.4f MB/s \t %.5f s\n", (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs );
#endif
    
#if(DEBUG_OUTPUT)
    printf("\nValidating GPU results...\n");
    printf(" ...reading back GPU results\n");
#endif
    checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
#if(DEBUG_OUTPUT)
    printf(" ...comparing the results\n");
#endif
    
	uint errorCount=0;
    for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
        if (h_HistogramGPU[i] != h_HistogramCPU[i])
        {
			if(errorCount++<10)
				printf("error: bin[%d]=%d expected:%d\n",i,h_HistogramGPU[i],h_HistogramCPU[i]);
            PassFailFlag = 0;
        }
#if(DEBUG_OUTPUT)
	printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");
#else
	if(!PassFailFlag)//print if fails
		printf(" ***256-bin histograms do not match!!!***\n\n");
#endif
    
#if(DEBUG_OUTPUT)
    printf("Shutting down 256-bin histogram...\n\n\n");
#endif
    closeHistogram256();
	//shuffled data deallocation
	checkCudaErrors(cudaFree(d_Shuffled_Data));
	return PassFailFlag;
}

int extension2Histogram(uchar *d_Data, uint  *d_Histogram, uint byteCount, uchar *h_Data, uint  *h_HistogramCPU, uint *h_HistogramGPU)
{
	int PassFailFlag = 1;
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
#if(DEBUG_OUTPUT)
    printf("Initializing 256-bin histogram...\n");
#endif
    initHistogram256();
#if(DEBUG_OUTPUT)
    printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);
#endif
    for (int iter = -1; iter < numRuns; iter++)
    {
        //iter == -1 -- warmup iteration
        if (iter == 0)
        {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }
        extension2Histogram(d_Histogram, d_Data, byteCount);
    }
    
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
#if(DEBUG_OUTPUT)
    printf("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
    printf("histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM256_THREADBLOCK_SIZE);
#else
	printf("Extension 2 histogram: %.4f MB/s \t %.5f s\n", (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs );
#endif
    
#if(DEBUG_OUTPUT)
    printf("\nValidating GPU results...\n");
    printf(" ...reading back GPU results\n");
#endif
    checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
#if(DEBUG_OUTPUT)
    printf(" ...comparing the results\n");
#endif
    
	uint errorCount=0;
    for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
        if (h_HistogramGPU[i] != h_HistogramCPU[i])
        {
			if(errorCount++<10)
				printf("error: bin[%d]=%d expected:%d\n",i,h_HistogramGPU[i],h_HistogramCPU[i]);
            PassFailFlag = 0;
        }
#if(DEBUG_OUTPUT)
	printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");
#else
	if(!PassFailFlag)//print if fails
		printf(" ***256-bin histograms do not match!!!***\n\n");
#endif
    
#if(DEBUG_OUTPUT)
    printf("Shutting down 256-bin histogram...\n\n\n");
#endif
    closeHistogram256();
	return PassFailFlag;
}

int extension3Histogram(uchar *d_Data, uint  *d_Histogram, uint byteCount, uchar *h_Data, uint  *h_HistogramCPU, uint *h_HistogramGPU)
{
	int PassFailFlag = 1;
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
#if(DEBUG_OUTPUT)
    printf("Initializing 256-bin histogram...\n");
#endif
    initHistogram256();
#if(DEBUG_OUTPUT)
    printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);
#endif
    for (int iter = -1; iter < numRuns; iter++)
    {
        //iter == -1 -- warmup iteration
        if (iter == 0)
        {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }
        extension3Histogram(d_Histogram, d_Data, byteCount);
    }
    
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
#if(DEBUG_OUTPUT)
    printf("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
    printf("histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM256_THREADBLOCK_SIZE);
#else
	printf("Extension 3 histogram: %.4f MB/s \t %.5f s\n", (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs );
#endif
    
#if(DEBUG_OUTPUT)
    printf("\nValidating GPU results...\n");
    printf(" ...reading back GPU results\n");
#endif
    checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
#if(DEBUG_OUTPUT)
    printf(" ...comparing the results\n");
#endif
	uint errorCount=0;
    for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
        if (h_HistogramGPU[i] != h_HistogramCPU[i])
        {
			if(errorCount++<10)
				printf("error: bin[%d]=%d expected:%d\n",i,h_HistogramGPU[i],h_HistogramCPU[i]);
            PassFailFlag = 0;
        }
#if(DEBUG_OUTPUT)
	printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");
#else
	if(!PassFailFlag)//print if fails
		printf(" ***256-bin histograms do not match!!!***\n\n");
#endif
    
#if(DEBUG_OUTPUT)
    printf("Shutting down 256-bin histogram...\n\n\n");
#endif
    closeHistogram256();
	return PassFailFlag;
}

int threadHistogram(uchar *d_Data, uint  *d_Histogram, uint byteCount, uchar *h_Data, uint  *h_HistogramCPU, uint *h_HistogramGPU)
{
	int PassFailFlag = 1;
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
#if(DEBUG_OUTPUT)
    printf("Initializing 256-bin histogram...\n");
#endif
    initHistogram256();
#if(DEBUG_OUTPUT)
    printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);
#endif
    for (int iter = -1; iter < numRuns; iter++)
    {
        //iter == -1 -- warmup iteration
        if (iter == 0)
        {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }
        threadHistogram(d_Histogram, d_Data, byteCount);
    }
    
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
#if(DEBUG_OUTPUT)
    printf("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
    printf("histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM256_THREADBLOCK_SIZE);
#else
	printf("Thread      histogram: %.4f MB/s \t %.5f s\n", (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs );
#endif
    
#if(DEBUG_OUTPUT)
    printf("\nValidating GPU results...\n");
    printf(" ...reading back GPU results\n");
#endif
    checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
#if(DEBUG_OUTPUT)
    printf(" ...comparing the results\n");
#endif
	uint errorCount=0;
    for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
        if (h_HistogramGPU[i] != h_HistogramCPU[i])
        {
			if(errorCount++<10)
				printf("error: bin[%d]=%d expected:%d\n",i,h_HistogramGPU[i],h_HistogramCPU[i]);
            PassFailFlag = 0;
        }
#if(DEBUG_OUTPUT)
	printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");
#else
	if(!PassFailFlag)//print if fails
		printf(" ***256-bin histograms do not match!!!***\n\n");
#endif
    
#if(DEBUG_OUTPUT)
    printf("Shutting down 256-bin histogram...\n\n\n");
#endif
    closeHistogram256();
	return PassFailFlag;
}

int threadPartialHistogram(uchar *d_Data, uint  *d_Histogram, uint byteCount, uchar *h_Data, uint  *h_HistogramCPU, uint *h_HistogramGPU)
{
	int PassFailFlag = 1;
	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
#if(DEBUG_OUTPUT)
    printf("Initializing 256-bin histogram...\n");
#endif
    initHistogram256();
#if(DEBUG_OUTPUT)
    printf("Running 256-bin GPU histogram for %u bytes (%u runs)...\n\n", byteCount, numRuns);
#endif
    for (int iter = -1; iter < numRuns; iter++)
    {
        //iter == -1 -- warmup iteration
        if (iter == 0)
        {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&hTimer);
            sdkStartTimer(&hTimer);
        }
        threadPartialHistogram(d_Histogram, d_Data, byteCount);
    }
    
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&hTimer);
    double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer) / (double)numRuns;
#if(DEBUG_OUTPUT)
    printf("histogram256() time (average) : %.5f sec, %.4f MB/sec\n\n", dAvgSecs, ((double)byteCount * 1.0e-6) / dAvgSecs);
    printf("histogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n",
           (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs, byteCount, 1, HISTOGRAM256_THREADBLOCK_SIZE);
#else
	printf("Thread      histogram: %.4f MB/s \t %.5f s\n", (1.0e-6 * (double)byteCount / dAvgSecs), dAvgSecs );
#endif
    
#if(DEBUG_OUTPUT)
    printf("\nValidating GPU results...\n");
    printf(" ...reading back GPU results\n");
#endif
    checkCudaErrors(cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost));
#if(DEBUG_OUTPUT)
    printf(" ...comparing the results\n");
#endif
	uint errorCount=0;
    for (uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
        if (h_HistogramGPU[i] != h_HistogramCPU[i])
        {
			if(errorCount++<10)
				printf("error: bin[%d]=%d expected:%d\n",i,h_HistogramGPU[i],h_HistogramCPU[i]);
            PassFailFlag = 0;
        }
#if(DEBUG_OUTPUT)
	printf(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n");
#else
	if(!PassFailFlag)//print if fails
		printf(" ***256-bin histograms do not match!!!***\n\n");
#endif
    
#if(DEBUG_OUTPUT)
    printf("Shutting down 256-bin histogram...\n\n\n");
#endif
    closeHistogram256();
	return PassFailFlag;
}

int testHistograms(uchar *d_Data, uint  *d_Histogram, uint byteCount, uchar *h_Data, uint  *h_HistogramCPU, uint *h_HistogramGPU)
{
	int result = 0;
	int passFlag = podlozhnyukHistogram(d_Data, d_Histogram, byteCount, h_Data, h_HistogramCPU, h_HistogramGPU);
	result = passFlag;
	passFlag = extension1Histogram(d_Data, d_Histogram, byteCount, h_Data, h_HistogramCPU, h_HistogramGPU);
	result = result << 1 | passFlag;
	passFlag = extension2Histogram(d_Data, d_Histogram, byteCount, h_Data, h_HistogramCPU, h_HistogramGPU);
	result = result << 1 | passFlag;
	passFlag = extension3Histogram(d_Data, d_Histogram, byteCount, h_Data, h_HistogramCPU, h_HistogramGPU);
    result = result << 1 | passFlag;
	passFlag = threadPartialHistogram(d_Data, d_Histogram, byteCount, h_Data, h_HistogramCPU, h_HistogramGPU);
    result = result << 1 | passFlag;
	return result;
}

int main(int argc, char **argv)
{
    uchar *h_Data;
    uint  *h_HistogramCPU, *h_HistogramGPU;
    uchar *d_Data;
    uint  *d_Histogram;
    uint byteCount = 64 * 1048576;

	printf("Number of runs: %d\n", numRuns);
	printf("Input size: %d\n", byteCount);
	
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;

#if(DEBUG_OUTPUT)
    printf("histogram - Starting...\n");
#endif

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
	
#if(DEBUG_OUTPUT)
    printf("Initializing data...\n");
    printf("...allocating CPU memory\n");
#endif
    h_Data         = (uchar *)malloc(byteCount);
    h_HistogramCPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
    h_HistogramGPU = (uint *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));

#if(DEBUG_OUTPUT)
    printf("...allocating GPU memory\n");
#endif
    checkCudaErrors(cudaMalloc((void **)&d_Data, byteCount));
    checkCudaErrors(cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)));
	

	
	{//input image: cosmos.jpg
		printf("Input: cosmos.jpg\n");
		loadDataFromImage(h_Data, byteCount, "cosmos.jpg");
	#if(DEBUG_OUTPUT)
		printf(" ...histogram256CPU()\n");
	#endif
		histogram256CPU(h_HistogramCPU, h_Data, byteCount);
	#if(DEBUG_OUTPUT)
		printf("...copying input data to GPU\n");
	#endif
		checkCudaErrors(cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));
		testHistograms(d_Data, d_Histogram, byteCount, h_Data, h_HistogramCPU, h_HistogramGPU);
	}

	{//input image: fruit.jpg
		printf("Input: fruit.jpg\n");
		loadDataFromImage(h_Data, byteCount, "fruit.jpg");
	#if(DEBUG_OUTPUT)
		printf(" ...histogram256CPU()\n");
	#endif
		histogram256CPU(h_HistogramCPU, h_Data, byteCount);
	#if(DEBUG_OUTPUT)
		printf("...copying input data to GPU\n");
	#endif
		checkCudaErrors(cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));
		testHistograms(d_Data, d_Histogram, byteCount, h_Data, h_HistogramCPU, h_HistogramGPU);
	}

	{//input image: forest.jpg
		printf("Input: forest.jpg\n");
		loadDataFromImage(h_Data, byteCount, "forest.jpg");
	#if(DEBUG_OUTPUT)
		printf(" ...histogram256CPU()\n");
	#endif
		histogram256CPU(h_HistogramCPU, h_Data, byteCount);
	#if(DEBUG_OUTPUT)
		printf("...copying input data to GPU\n");
	#endif
		checkCudaErrors(cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));
		testHistograms(d_Data, d_Histogram, byteCount, h_Data, h_HistogramCPU, h_HistogramGPU);
	}

	{//input image: rabbits.jpg
		printf("Input: rabbits.jpg\n");
		loadDataFromImage(h_Data, byteCount, "rabbits.jpg");
	#if(DEBUG_OUTPUT)
		printf(" ...histogram256CPU()\n");
	#endif
		histogram256CPU(h_HistogramCPU, h_Data, byteCount);
	#if(DEBUG_OUTPUT)
		printf("...copying input data to GPU\n");
	#endif
		checkCudaErrors(cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));
		testHistograms(d_Data, d_Histogram, byteCount, h_Data, h_HistogramCPU, h_HistogramGPU);
	}
	
	{//input random
		printf("Input: random\n");
		generateRandomData(h_Data, byteCount);
	#if(DEBUG_OUTPUT)
		printf(" ...histogram256CPU()\n");
	#endif
		histogram256CPU(h_HistogramCPU, h_Data, byteCount);
	#if(DEBUG_OUTPUT)
		printf("...copying input data to GPU\n");
	#endif
		checkCudaErrors(cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));
		testHistograms(d_Data, d_Histogram, byteCount, h_Data, h_HistogramCPU, h_HistogramGPU);
	}
	
	{//input degenerate
		printf("Input: degenerate\n");
		generateDegenerateData(h_Data, byteCount);
	#if(DEBUG_OUTPUT)
		printf(" ...histogram256CPU()\n");
	#endif
		histogram256CPU(h_HistogramCPU, h_Data, byteCount);
	#if(DEBUG_OUTPUT)
		printf("...copying input data to GPU\n");
	#endif
		checkCudaErrors(cudaMemcpy(d_Data, h_Data, byteCount, cudaMemcpyHostToDevice));
		testHistograms(d_Data, d_Histogram, byteCount, h_Data, h_HistogramCPU, h_HistogramGPU);
	}

#if(DEBUG_OUTPUT)
    printf("Shutting down...\n");
#endif
    checkCudaErrors(cudaFree(d_Histogram));
    checkCudaErrors(cudaFree(d_Data));
    free(h_HistogramGPU);
    free(h_HistogramCPU);
    free(h_Data);

    cudaDeviceReset();
#if(DEBUG_OUTPUT)
    printf("histogram - Test Summary\n");
    printf("Test passed\n");
#endif
    exit(EXIT_SUCCESS);
}
