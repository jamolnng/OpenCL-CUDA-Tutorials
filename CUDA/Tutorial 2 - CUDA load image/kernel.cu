#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "PNG.h"

__global__ void copy(const unsigned char* in, unsigned char* out)
{
	int x = blockIdx.x;
	int y = threadIdx.x;
	int width = blockDim.x;
	int index = (x + y * width) * 4;

	//copy each color channel
	out[index] = in[index];
	out[index + 1] = in[index + 1];
	out[index + 2] = in[index + 2];
	out[index + 3] = in[index + 3];
}

int main(int arg, char* args[])
{
	PNG inPng("Lenna.png");
	PNG outPng;
	outPng.Create(inPng.w, inPng.h);

	//store width and height so we can use them for our output image later
	const unsigned int w = inPng.w;
	const unsigned int h = inPng.h;
	//4 because there are 4 color channels R, G, B, and A
	int size = w * h * 4;

	unsigned char *in = 0;
	unsigned char *out = 0;
	cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
	{
		std::cout << "No CUDA devices found!" << std::endl;
		exit(1);
    }

	//prints the device the kernel will be running on
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout << "Using device: " << prop.name << std::endl;
	
    // Allocate GPU buffers for the images
	cudaMalloc((void**)&in, size * sizeof(unsigned char));
	cudaMalloc((void**)&out, size * sizeof(unsigned char));
	
    // Copy image data from host memory to GPU buffers.
    cudaMemcpy(in, &inPng.data[0], size * sizeof(unsigned char), cudaMemcpyHostToDevice);

	//free the input image because we do not need it anymore
	inPng.Free();

	// Launch a kernel on the GPU with one thread for each element.
	copy<<<w, h>>>(in, out);

	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
	{
		std::cout << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		cudaFree(in);
		cudaFree(out);
		exit(1);
    }
	
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
	{
		std::cout << "Could not synchronize device!" << std::endl;
		cudaFree(in);
		cudaFree(out);
		exit(1);
    }

	//temporary array to store the result from opencl
	auto tmp = new unsigned char[w * h * 4];
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(tmp, out, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaFree(in);
	cudaFree(out);

	//copy the data from the temp array to the png
	std::copy(&tmp[0], &tmp[w * h * 4], std::back_inserter(outPng.data));

	//write the image to file
	outPng.Save("cuda_tutorial_2.png");
	//free the iamge's resources since we are done with it
	outPng.Free();

	//free the temp array
	delete[] tmp;

	if(cudaStatus != cudaSuccess)
	{
		std::cout << "Could not copy buffer memory to host!" << std::endl;
		exit(1);
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
	{
		std::cout << "Device reset failed!" << std::endl;
        exit(1);
    }

    return 0;
}