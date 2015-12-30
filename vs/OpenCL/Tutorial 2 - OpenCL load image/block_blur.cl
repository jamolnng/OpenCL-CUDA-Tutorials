const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

void kernel block_blur(image2d_t in, image2d_t out)
{
	for(int x = 0; x < 512; x++)
	{
		for(int y = 0; y < 512; y++)
		{
			int2 pos = (int2)(x, y);
			uint4 bgra = read_imageui(in, smp, pos);
			write_imageui(out, pos, bgra);
		}
	}
}