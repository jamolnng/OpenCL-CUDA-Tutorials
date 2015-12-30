#pragma once

#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

//#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__) || defined(__MINGW32__) || defined(__MINGW64__) || defined(WIN32_LEAN_AND_MEAN)
//#include <Windows.h>
//typedef struct tagBITMAPFILEHEADER
//{
//	WORD  bfType;
//	DWORD bfSize;
//	WORD  bfReserved1;
//	WORD  bfReserved2;
//	DWORD bfOffBits;
//} BITMAPFILEHEADER, *PBITMAPFILEHEADER;
//#else
////linux stuff to be here
//#endif

#pragma pack(2)

typedef struct
{
	unsigned short bfType;
	unsigned int   bfSize;
	unsigned short bfReserved1;
	unsigned short bfReserved2;
	unsigned int   bfOffBits;
} BITMAPFILEHEADER;

#pragma pack()

#  define BF_TYPE 0x4D42

typedef struct
{
	unsigned int   biSize;
	int            biWidth;
	int            biHeight;
	unsigned short biPlanes;
	unsigned short biBitCount;
	unsigned int   biCompression;
	unsigned int   biSizeImage;
	int            biXPelsPerMeter;
	int            biYPelsPerMeter;
	unsigned int   biClrUsed;
	unsigned int   biClrImportant;
} BITMAPINFOHEADER;

using Uint8 = uint8_t;

class Bitmap
{
public:
	size_t w, h, s;
	Uint8* pixels;

	Bitmap() {}
	Bitmap(std::string fileName)
	{
		Load(fileName);
	}

	int Create(size_t w, size_t h)
	{
		this->w = w;
		this->h = h;
		s = w * h * 3;
		this->pixels = new Uint8[s];
		return 0;
	}

	int Load(std::string fileName)
	{
		Uint8* datBuff[2] = { nullptr, nullptr }; // Header buffers
		BITMAPFILEHEADER* bmpHeader = nullptr; // Header
		BITMAPINFOHEADER* bmpInfo = nullptr; // Info

		std::ifstream file(fileName, std::ios::binary);
		if (!file)
		{
			std::cout << "Failure to open bitmap file.\n";
			return 1;
		}

		datBuff[0] = new Uint8[sizeof(BITMAPFILEHEADER)];
		datBuff[1] = new Uint8[sizeof(BITMAPINFOHEADER)];

		file.read((char*)datBuff[0], sizeof(BITMAPFILEHEADER));
		file.read((char*)datBuff[1], sizeof(BITMAPINFOHEADER));

		bmpHeader = (BITMAPFILEHEADER*)datBuff[0];
		bmpInfo = (BITMAPINFOHEADER*)datBuff[1];

		//make sure it is actually a bitmap file
		if (bmpHeader->bfType != BF_TYPE)
		{
			std::cout << "File \"" << fileName << "\" isn't a bitmap file\n";
			Free();
			return 2;
		}

		s = bmpInfo->biSizeImage;

		pixels = new Uint8[s];

		// Go to where image data starts, then read in image data
		file.seekg(bmpHeader->bfOffBits);
		file.read((char*)pixels, bmpInfo->biSizeImage);
		file.close();

		//Uint8 tmpRGB = 0; // Swap buffer
		//for (unsigned long i = 0; i < bmpInfo->biSizeImage; i += 3)
		//{
		//	tmpRGB = pixels[i];
		//	pixels[i] = pixels[i + 2];
		//	pixels[i + 2] = tmpRGB;
		//}

		// Set width and height to the values loaded from the file
		w = bmpInfo->biWidth;
		h = bmpInfo->biHeight;

		delete bmpHeader;
		delete bmpInfo;

		//delete datBuff[0];
		//delete datBuff[1];

		return 0;
	}

	void Free()
	{
		w = 0;
		h = 0;
		s = 0;
		delete[] pixels;
		pixels = NULL;
	}

	int Save(std::string fileName)
	{
		return Save(fileName, w, h, pixels);
	}

	static int Save(std::string fileName, size_t w, size_t h, Uint8* pixels)
	{
		if (pixels == NULL) return 1;

		size_t s = w * h * 3;

		BITMAPFILEHEADER BitmapFileHeader =
		{
			BF_TYPE, //Bmp Mark
			sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + 3 * w * h, //File Size
			0,    //Not Used
			0,    //Not Used
			sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER)    //Colors Offset in File
		};

		BITMAPINFOHEADER BitmapInfoHeader =
		{
			sizeof(BITMAPINFOHEADER),    //Structure Size
			w,
			h,
			1,        //Number of Plans
			24,        //Bits Per colors 
			0,        //Compression Ration
			0,        //Original Image Size (When Compressed)
			2835,    //Number of Pixel Per Meter (Vertical)
			2835,    //Number of Pixel Per Meter (Horizontal)
			0,        //Number of Used Colors (Indexed Mode)
			0,        //Number of Important Colors (Indexed Mode)
		};

		//Uint8 tmpRGB = 0; // Swap buffer
		//for (unsigned long i = 0; i < s; i += 3)
		//{
		//	tmpRGB = pixels[i];
		//	pixels[i] = pixels[i + 2];
		//	pixels[i + 2] = tmpRGB;
		//}

		std::ofstream outfile(fileName, std::ios::binary);
		outfile.write((char*)&BitmapFileHeader, sizeof(BitmapFileHeader));
		outfile.write((char*)&BitmapInfoHeader, sizeof(BitmapInfoHeader));
		outfile.write((char*)pixels, 3 * w * h);
		outfile.close();

		return 0;
	}
};