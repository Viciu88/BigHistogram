#ifndef HISTOGRAM_H
#define HISTOGRAM_H

extern "C" void approxHistogramGPU(uint *d_Histogram, void *d_Data, uint byteCount, uint binCount, cudaDeviceProp deviceProp);
extern "C" void closePartialHistograms(void);
extern "C" void initPartialHistograms(uint partialHistogramCount, uint binCount);
extern "C" void baseHistogramGPU(uint *d_Histogram, void *d_Data, uint byteCount, uint binCount, cudaDeviceProp deviceProp);
extern "C" void baseHistogramAtomicGPU(uint *d_Histogram, void *d_Data, uint byteCount, uint binCount, cudaDeviceProp deviceProp);

#endif