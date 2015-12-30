const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

void kernel copy(__read_only image2d_t in, __write_only image2d_t out)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int2 pos = (int2)(x, y);
	uint4 pixel = read_imageui(in, smp, pos);
	write_imageui(out, pos, pixel);
}