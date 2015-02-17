all: 
	nvcc histogram256.cu histogram64.cu histogram_gold.cpp main.cpp -o histogram -gencode arch=compute_20,code=sm_20 -I/usr/local/cuda/samples/common/inc -L/usr/local/lib -lopencv_core -lopencv_highgui -I/usr/local/include/opencv

