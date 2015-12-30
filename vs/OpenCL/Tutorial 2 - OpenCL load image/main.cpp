//#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include "Bitmap.h"

std::string readFile(std::string fileName)
{
	std::ifstream t(fileName);
	std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
	return str;
}

int main(int arg, char* args[])
{

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
	platforms[1].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	//Set the device to the first device in the platform. You can have more than one device associated with a single platform, for instance if you had two of the same GPUs on your system in SLI or CrossFire.
	cl::Device device = devices[0];

	//This is just helpful to see what device and platform you are using.
	std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	std::cout << "Using platform: " << platforms[1].getInfo<CL_PLATFORM_NAME>() << std::endl;

	//Finally create the OpenCL context from the device you have chosen.
	const cl::Context context(device);

	//load our image
	Bitmap b("Lenna.bmp");

	cl_int err = 0;

	const cl::ImageFormat format(CL_RGB, CL_UNSIGNED_INT8);
	cl::Image2D in(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, b.w, b.h, 0, b.pixels, &err);

	//create out output image
	cl::Image2D out(context, CL_MEM_WRITE_ONLY, format, b.w, b.h, 0, NULL, &err);

	cl::Program::Sources sources;
	std::string kernel_code = readFile("block_blur.cl");
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

	cl::Kernel block_blur(program, "block_blur");
	//Set our arguements for the kernel
	block_blur.setArg(0, in);
	block_blur.setArg(1, out);
	//block_blur.setArg(2, 10);

	queue.finish();

	cl::Event e;
	queue.enqueueNDRangeKernel(block_blur, cl::NullRange, cl::NDRange(1), cl::NullRange, 0, &e);
	queue.finish();
	e.wait();

	cl::size_t<3> origin;
	cl::size_t<3> size;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;

	size[0] = b.w;
	size[1] = b.h;
	size[2] = 1;

	Uint8* pout = new Uint8[b.w * b.h * 3];
	queue.enqueueReadImage(out, CL_TRUE, origin, size, 0, 0, pout, 0, &e);
	e.wait();
	queue.finish();

	Bitmap::Save("out.bmp", b.w, b.h, pout);

	b.Free();

	return 0;
}