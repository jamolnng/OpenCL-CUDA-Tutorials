//#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <string>
#include <sstream>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "PNG.h"

std::string readFile(std::string fileName)
{
	std::ifstream t(fileName);
	std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
	return str;
}

int main(int arg, char* args[])
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0)
	{
		std::cout << "No OpenCL platforms found" << std::endl;//This means you do not have an OpenCL compatible platform on your system.
		exit(1);
	}
	std::vector<cl::Device> devices;
	platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	cl::Device device = devices[0];
	std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "Using platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;
	cl::Context context(device);

	//load our image
	PNG inPng("Lenna.png");

	//store width and height so we can use them for our output image later
	const unsigned int w = inPng.w;
	const unsigned int h = inPng.h;

	//input image
	const cl::ImageFormat format(CL_RGBA, CL_UNSIGNED_INT8);
	cl::Image2D in(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, w, h, 0, &inPng.data[0]);

	//we are done with the image so free up its memory
	inPng.Free();

	//output image
	cl::Image2D out(context, CL_MEM_WRITE_ONLY, format, w, h, 0, NULL);

	cl::Program::Sources sources;
	std::string kernel_code = readFile("cl_tutorial_2_copy.cl");
	//Add your program source
	sources.push_back({ kernel_code.c_str(),kernel_code.length() });

	//Create your OpenCL program and build it.
	cl::Program program(context, sources);
	if (program.build({ device }) != CL_SUCCESS)
	{
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;//print the build log to find any issues with your source
		exit(1);//Quit if your program doesn't compile
	}

	//set the kernel arguments
	cl::Kernel kernelCopy(program, "copy");
	kernelCopy.setArg(0, in);
	kernelCopy.setArg(1, out);

	//create command queue
	cl::CommandQueue queue(context, device, 0, NULL);

	//execute kernel
	queue.enqueueNDRangeKernel(kernelCopy, cl::NullRange, cl::NDRange(w, h), cl::NullRange);

	//wait for kernel to finish
	queue.finish();

	//start and end coordinates for reading our image (I really do not like how the c++ wrapper does this)
	cl::size_t<3> origin;
	cl::size_t<3> size;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;
	size[0] = w;
	size[1] = h;
	size[2] = 1;

	//output png
	PNG outPng;
	//create the image with the same width and height as original
	outPng.Create(w, h);

	//temporary array to store the result from opencl
	auto tmp = new unsigned char[w * h * 4];
	//CL_TRUE means that it waits for the entire image to be copied before continuing
	queue.enqueueReadImage(out, CL_TRUE, origin, size, 0, 0, tmp);

	//copy the data from the temp array to the png
	std::copy(&tmp[0], &tmp[w * h * 4], std::back_inserter(outPng.data));

	//write the image to file
	outPng.Save("cl_tutorial_2.png");
	//free the iamge's resources since we are done with it
	outPng.Free();

	//free the temp array
	delete[] tmp;

	return 0;
}