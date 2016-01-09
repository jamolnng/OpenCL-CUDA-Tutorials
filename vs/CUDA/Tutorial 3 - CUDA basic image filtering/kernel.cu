#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "PNG.h"

__global__ void boxFilter(const unsigned char* in, unsigned char* out, const int imageWidth, const int imageHeight, const int halfBoxWidth, const int halfBoxHeight)
{
	int x = blockIdx.x;
	int y = blockIdx.y;

	int count = 0;

	int index = (x + y * imageWidth) * 4;

	unsigned int total[4] = { 0, 0, 0, 0 };

	for (int i = -halfBoxWidth; i <= halfBoxWidth; i++)
	{
		for (int j = -halfBoxHeight; j <= halfBoxHeight; j++)
		{
			int cx = x + i;
			int cy = y + j;
			if (cx >= 0 && cy >= 0 && cx < imageWidth && cy < imageHeight)
			{
				int adjIndex = (cx + cy * imageWidth) * 4;
				for (int c = 0; c < 4; c++)
				{
					total[c] += static_cast<unsigned int>(in[adjIndex + c]);
				}
				count++;
			}
		}
	}

	out[index]     = static_cast<unsigned char>(total[0] / count);
	out[index + 1] = static_cast<unsigned char>(total[1] / count);
	out[index + 2] = static_cast<unsigned char>(total[2] / count);
	out[index + 3] = static_cast<unsigned char>(total[3] / count);
}

int main(int arg, char* args[])
{
	int filterWidth = 10;
	int filterHeight = 10;
	if (arg > 2)
	{
		filterWidth = std::atoi(args[1]);
		filterHeight = std::atoi(args[2]);
	}

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
	dim3 block_size(w, h);
	dim3 grid_size(1);
	boxFilter<<<block_size, 1>>>(in, out, w, h, filterWidth, filterHeight);

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
	outPng.Save("cuda_tutorial_3.png");
	//free the iamge's resources since we are done with it
	outPng.Free();

	//free the temp array
	delete[] tmp;

	if (cudaStatus != cudaSuccess)
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