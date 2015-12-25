#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>

std::string readFile(std::string fileName)
{
	std::ifstream t(fileName);
	std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
	return str;
}

int main(int arg, char* args[])
{
	const int size = 10;
	int A[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int B[] = { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };
	int C[size];

	//stl vector to store all of the available platforms
	std::vector<cl::Platform> platforms;
	//get all available platforms
	cl::Platform::get(&platforms);

	if (platforms.size() == 0)
	{
		std::cout << "No OpenCL platforms found" << std::endl;//This means you do not have an OpenCL compatible platform on your system.
		exit(1);
	}

	//Create a stl vector to store all of the availbe devices to use from the first platform.
	std::vector<cl::Device> devices;
	//Get the available devices from the platform. For me the platform for my 980ti is actually th e second in the platform list but for simplicity we will use the first one.
	platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	//Set the device to the first device in the platform. You can have more than one device associated with a single platform, for instance if you had two of the same GPUs on your system in SLI or CrossFire.
	cl::Device device = devices[0];

	//This is just helpful to see what device and platform you are using.
	std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "Using platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;

	//Finally create the OpenCL context from the device you have chosen.
	cl::Context context(device);

	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * size);
	cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * size);

	//A source object for your program
	cl::Program::Sources sources;
	std::string kernel_code = readFile("simple_add.cl");
	//Add your program source
	sources.push_back({ kernel_code.c_str(),kernel_code.length() });

	//Create your OpenCL program and build it.
	cl::Program program(context, sources);
	if (program.build({ device }) != CL_SUCCESS)
	{
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;//print the build log to find any issues with your source
		exit(1);//Quit if your program doesn't compile
	}

	cl::CommandQueue queue(context, device, 0, NULL);

	//Write our buffers that we are adding to our OpenCL device
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * size, A);
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * size, B);

	//Create our Kernel (basically what is the starting point for our OpenCL program)
	cl::Kernel simple_add(program, "simple_add");
	//Set our arguements for the kernel
	simple_add.setArg(0, buffer_A);
	simple_add.setArg(1, buffer_B);
	simple_add.setArg(2, buffer_C);

	//Make sure that our queue is done with all of its tasks before continuing
	queue.finish();

	//Create an event that we can use to wait for our program to finish running
	cl::Event e;
	//This runs our program, the ranges here are the offset, global, local ranges that our code runs in.
	queue.enqueueNDRangeKernel(simple_add, cl::NullRange, cl::NDRange(size), cl::NullRange, 0, &e);

	//Waits for our program to finish
	e.wait();
	//Reads the output written to our buffer into our final array
	queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * size, C);

	//prints the array
	std::cout << "Result:" << std::endl;
	for (int i = 0; i < size; i++)
	{
		std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
	}

	return 0;
}