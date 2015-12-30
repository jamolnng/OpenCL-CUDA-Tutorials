#pragma once

#include <vector>
#include <string>

#include "lodepng.h"

class PNG
{
public:
	size_t w, h;
	std::vector<unsigned char> data;

	PNG() {}
	PNG(std::string file)
	{
		Load(file);
	}

	unsigned Load(std::string file)
	{
		Free();
		return lodepng::decode(data, w, h, file.c_str());
	}

	unsigned Save(std::string file)
	{
		return lodepng::encode(file.c_str(), data, w, h);
	}

	void Create(size_t w, size_t h)
	{
		Free();
		this->w = w;
		this->h = h;
		data.reserve(w * h * 4);
	}

	void Free()
	{
		w = 0;
		h = 0;
		data.clear();
	}
};