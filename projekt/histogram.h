#ifndef HISTOGRAM_H
#define HISTOGRAM_H

extern "C" void histogramGPU(uint *d_Histogram, void *d_Data, uint byteCount, uint binCount, cudaDeviceProp deviceProp);
extern "C" void closePartialHistograms(void);
extern "C" void initPartialHistograms(uint partialHistogramCount, uint binCount);

#endif