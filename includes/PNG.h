#pragma once

#include <vector>
#include <string>
#ifndef LODEPNG_H
#define LODEPNG_H 
#include <string.h>
#ifdef __cplusplus
#include <vector>
#include <string>
#endif
extern const char* LODEPNG_VERSION_STRING;
#ifndef LODEPNG_NO_COMPILE_ZLIB
#define LODEPNG_COMPILE_ZLIB 
#endif
#ifndef LODEPNG_NO_COMPILE_PNG
#define LODEPNG_COMPILE_PNG 
#endif
#ifndef LODEPNG_NO_COMPILE_DECODER
#define LODEPNG_COMPILE_DECODER 
#endif
#ifndef LODEPNG_NO_COMPILE_ENCODER
#define LODEPNG_COMPILE_ENCODER 
#endif
#ifndef LODEPNG_NO_COMPILE_DISK
#define LODEPNG_COMPILE_DISK 
#endif
#ifndef LODEPNG_NO_COMPILE_ANCILLARY_CHUNKS
#define LODEPNG_COMPILE_ANCILLARY_CHUNKS 
#endif
#ifndef LODEPNG_NO_COMPILE_ERROR_TEXT
#define LODEPNG_COMPILE_ERROR_TEXT 
#endif
#ifndef LODEPNG_NO_COMPILE_ALLOCATORS
#define LODEPNG_COMPILE_ALLOCATORS 
#endif
#ifdef __cplusplus
#ifndef LODEPNG_NO_COMPILE_CPP
#define LODEPNG_COMPILE_CPP 
#endif
#endif
#ifdef LODEPNG_COMPILE_PNG
typedef enum LodePNGColorType
{
	LCT_GREY = 0,
	LCT_RGB = 2,
	LCT_PALETTE = 3,
	LCT_GREY_ALPHA = 4,
	LCT_RGBA = 6
} LodePNGColorType;
#ifdef LODEPNG_COMPILE_DECODER
unsigned lodepng_decode_memory(unsigned char** out, unsigned* w, unsigned* h, const unsigned char* in, size_t insize, LodePNGColorType colortype, unsigned bitdepth);
unsigned lodepng_decode32(unsigned char** out, unsigned* w, unsigned* h, const unsigned char* in, size_t insize);
unsigned lodepng_decode24(unsigned char** out, unsigned* w, unsigned* h, const unsigned char* in, size_t insize);
#ifdef LODEPNG_COMPILE_DISK
unsigned lodepng_decode_file(unsigned char** out, unsigned* w, unsigned* h, const char* filename, LodePNGColorType colortype, unsigned bitdepth);
unsigned lodepng_decode32_file(unsigned char** out, unsigned* w, unsigned* h, const char* filename);
unsigned lodepng_decode24_file(unsigned char** out, unsigned* w, unsigned* h, const char* filename);
#endif
#endif
#ifdef LODEPNG_COMPILE_ENCODER
unsigned lodepng_encode_memory(unsigned char** out, size_t* outsize, const unsigned char* image, unsigned w, unsigned h, LodePNGColorType colortype, unsigned bitdepth);
unsigned lodepng_encode32(unsigned char** out, size_t* outsize, const unsigned char* image, unsigned w, unsigned h);
unsigned lodepng_encode24(unsigned char** out, size_t* outsize, const unsigned char* image, unsigned w, unsigned h);
#ifdef LODEPNG_COMPILE_DISK
unsigned lodepng_encode_file(const char* filename, const unsigned char* image, unsigned w, unsigned h, LodePNGColorType colortype, unsigned bitdepth);
unsigned lodepng_encode32_file(const char* filename, const unsigned char* image, unsigned w, unsigned h);
unsigned lodepng_encode24_file(const char* filename, const unsigned char* image, unsigned w, unsigned h);
#endif
#endif
#ifdef LODEPNG_COMPILE_CPP
namespace lodepng
{
#ifdef LODEPNG_COMPILE_DECODER
	unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h, const unsigned char* in, size_t insize, LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
	unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h, const std::vector<unsigned char>& in, LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
#ifdef LODEPNG_COMPILE_DISK
	unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h, const std::string& filename, LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
#endif
#endif
#ifdef LODEPNG_COMPILE_ENCODER
	unsigned encode(std::vector<unsigned char>& out, const unsigned char* in, unsigned w, unsigned h, LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
	unsigned encode(std::vector<unsigned char>& out, const std::vector<unsigned char>& in, unsigned w, unsigned h, LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
#ifdef LODEPNG_COMPILE_DISK
	unsigned encode(const std::string& filename, const unsigned char* in, unsigned w, unsigned h, LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
	unsigned encode(const std::string& filename, const std::vector<unsigned char>& in, unsigned w, unsigned h, LodePNGColorType colortype = LCT_RGBA, unsigned bitdepth = 8);
#endif
#endif
}
#endif
#endif
#ifdef LODEPNG_COMPILE_ERROR_TEXT
const char* lodepng_error_text(unsigned code);
#endif
#ifdef LODEPNG_COMPILE_DECODER
typedef struct LodePNGDecompressSettings LodePNGDecompressSettings;
struct LodePNGDecompressSettings
{
	unsigned ignore_adler32;
	unsigned(*custom_zlib)(unsigned char**, size_t*, const unsigned char*, size_t, const LodePNGDecompressSettings*);
	unsigned(*custom_inflate)(unsigned char**, size_t*, const unsigned char*, size_t, const LodePNGDecompressSettings*);
	const void* custom_context;
};
extern const LodePNGDecompressSettings lodepng_default_decompress_settings;
void lodepng_decompress_settings_init(LodePNGDecompressSettings* settings);
#endif
#ifdef LODEPNG_COMPILE_ENCODER
typedef struct LodePNGCompressSettings LodePNGCompressSettings;
struct LodePNGCompressSettings
{
	unsigned btype;
	unsigned use_lz77;
	unsigned windowsize;
	unsigned minmatch;
	unsigned nicematch;
	unsigned lazymatching;
	unsigned(*custom_zlib)(unsigned char**, size_t*, const unsigned char*, size_t, const LodePNGCompressSettings*);
	unsigned(*custom_deflate)(unsigned char**, size_t*, const unsigned char*, size_t, const LodePNGCompressSettings*);
	const void* custom_context;
};
extern const LodePNGCompressSettings lodepng_default_compress_settings;
void lodepng_compress_settings_init(LodePNGCompressSettings* settings);
#endif
#ifdef LODEPNG_COMPILE_PNG
typedef struct LodePNGColorMode
{
	LodePNGColorType colortype;
	unsigned bitdepth;
	unsigned char* palette;
	size_t palettesize;
	unsigned key_defined;
	unsigned key_r;
	unsigned key_g;
	unsigned key_b;
} LodePNGColorMode;
void lodepng_color_mode_init(LodePNGColorMode* info);
void lodepng_color_mode_cleanup(LodePNGColorMode* info);
unsigned lodepng_color_mode_copy(LodePNGColorMode* dest, const LodePNGColorMode* source);
void lodepng_palette_clear(LodePNGColorMode* info);
unsigned lodepng_palette_add(LodePNGColorMode* info, unsigned char r, unsigned char g, unsigned char b, unsigned char a);
unsigned lodepng_get_bpp(const LodePNGColorMode* info);
unsigned lodepng_get_channels(const LodePNGColorMode* info);
unsigned lodepng_is_greyscale_type(const LodePNGColorMode* info);
unsigned lodepng_is_alpha_type(const LodePNGColorMode* info);
unsigned lodepng_is_palette_type(const LodePNGColorMode* info);
unsigned lodepng_has_palette_alpha(const LodePNGColorMode* info);
unsigned lodepng_can_have_alpha(const LodePNGColorMode* info);
size_t lodepng_get_raw_size(unsigned w, unsigned h, const LodePNGColorMode* color);
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
typedef struct LodePNGTime
{
	unsigned year;
	unsigned month;
	unsigned day;
	unsigned hour;
	unsigned minute;
	unsigned second;
} LodePNGTime;
#endif
typedef struct LodePNGInfo
{
	unsigned compression_method;
	unsigned filter_method;
	unsigned interlace_method;
	LodePNGColorMode color;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
	unsigned background_defined;
	unsigned background_r;
	unsigned background_g;
	unsigned background_b;
	size_t text_num;
	char** text_keys;
	char** text_strings;
	size_t itext_num;
	char** itext_keys;
	char** itext_langtags;
	char** itext_transkeys;
	char** itext_strings;
	unsigned time_defined;
	LodePNGTime time;
	unsigned phys_defined;
	unsigned phys_x;
	unsigned phys_y;
	unsigned phys_unit;
	unsigned char* unknown_chunks_data[3];
	size_t unknown_chunks_size[3];
#endif
} LodePNGInfo;
void lodepng_info_init(LodePNGInfo* info);
void lodepng_info_cleanup(LodePNGInfo* info);
unsigned lodepng_info_copy(LodePNGInfo* dest, const LodePNGInfo* source);
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
void lodepng_clear_text(LodePNGInfo* info);
unsigned lodepng_add_text(LodePNGInfo* info, const char* key, const char* str);
void lodepng_clear_itext(LodePNGInfo* info);
unsigned lodepng_add_itext(LodePNGInfo* info, const char* key, const char* langtag, const char* transkey, const char* str);
#endif
unsigned lodepng_convert(unsigned char* out, const unsigned char* in, const LodePNGColorMode* mode_out, const LodePNGColorMode* mode_in, unsigned w, unsigned h);
#ifdef LODEPNG_COMPILE_DECODER
typedef struct LodePNGDecoderSettings
{
	LodePNGDecompressSettings zlibsettings;
	unsigned ignore_crc;
	unsigned color_convert;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
	unsigned read_text_chunks;
	unsigned remember_unknown_chunks;
#endif
} LodePNGDecoderSettings;
void lodepng_decoder_settings_init(LodePNGDecoderSettings* settings);
#endif
#ifdef LODEPNG_COMPILE_ENCODER
typedef enum LodePNGFilterStrategy
{
	LFS_ZERO,
	LFS_MINSUM,
	LFS_ENTROPY,
	LFS_BRUTE_FORCE,
	LFS_PREDEFINED
} LodePNGFilterStrategy;
typedef struct LodePNGColorProfile
{
	unsigned colored;
	unsigned key;
	unsigned short key_r;
	unsigned short key_g;
	unsigned short key_b;
	unsigned alpha;
	unsigned numcolors;
	unsigned char palette[1024];
	unsigned bits;
} LodePNGColorProfile;
void lodepng_color_profile_init(LodePNGColorProfile* profile);
unsigned lodepng_get_color_profile(LodePNGColorProfile* profile, const unsigned char* image, unsigned w, unsigned h, const LodePNGColorMode* mode_in);
unsigned lodepng_auto_choose_color(LodePNGColorMode* mode_out, const unsigned char* image, unsigned w, unsigned h, const LodePNGColorMode* mode_in);
typedef struct LodePNGEncoderSettings
{
	LodePNGCompressSettings zlibsettings;
	unsigned auto_convert;
	unsigned filter_palette_zero;
	LodePNGFilterStrategy filter_strategy;
	const unsigned char* predefined_filters;
	unsigned force_palette;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
	unsigned add_id;
	unsigned text_compression;
#endif
} LodePNGEncoderSettings;

void lodepng_encoder_settings_init(LodePNGEncoderSettings* settings);
#endif


#if defined(LODEPNG_COMPILE_DECODER) || defined(LODEPNG_COMPILE_ENCODER)

typedef struct LodePNGState
{
#ifdef LODEPNG_COMPILE_DECODER
	LodePNGDecoderSettings decoder;
#endif
#ifdef LODEPNG_COMPILE_ENCODER
	LodePNGEncoderSettings encoder;
#endif
	LodePNGColorMode info_raw;
	LodePNGInfo info_png;
	unsigned error;
#ifdef LODEPNG_COMPILE_CPP

	virtual ~LodePNGState() {}
#endif
} LodePNGState;
void lodepng_state_init(LodePNGState* state);
void lodepng_state_cleanup(LodePNGState* state);
void lodepng_state_copy(LodePNGState* dest, const LodePNGState* source);
#endif
#ifdef LODEPNG_COMPILE_DECODER
unsigned lodepng_decode(unsigned char** out, unsigned* w, unsigned* h, LodePNGState* state, const unsigned char* in, size_t insize);
unsigned lodepng_inspect(unsigned* w, unsigned* h, LodePNGState* state, const unsigned char* in, size_t insize);
#endif
#ifdef LODEPNG_COMPILE_ENCODER
unsigned lodepng_encode(unsigned char** out, size_t* outsize, const unsigned char* image, unsigned w, unsigned h, LodePNGState* state);
#endif
unsigned lodepng_chunk_length(const unsigned char* chunk);
void lodepng_chunk_type(char type[5], const unsigned char* chunk);
unsigned char lodepng_chunk_type_equals(const unsigned char* chunk, const char* type);
unsigned char lodepng_chunk_ancillary(const unsigned char* chunk);
unsigned char lodepng_chunk_private(const unsigned char* chunk);
unsigned char lodepng_chunk_safetocopy(const unsigned char* chunk);
unsigned char* lodepng_chunk_data(unsigned char* chunk);
const unsigned char* lodepng_chunk_data_const(const unsigned char* chunk);
unsigned lodepng_chunk_check_crc(const unsigned char* chunk);
void lodepng_chunk_generate_crc(unsigned char* chunk);
unsigned char* lodepng_chunk_next(unsigned char* chunk);
const unsigned char* lodepng_chunk_next_const(const unsigned char* chunk);
unsigned lodepng_chunk_append(unsigned char** out, size_t* outlength, const unsigned char* chunk);
unsigned lodepng_chunk_create(unsigned char** out, size_t* outlength, unsigned length, const char* type, const unsigned char* data);
unsigned lodepng_crc32(const unsigned char* buf, size_t len);
#endif
#ifdef LODEPNG_COMPILE_ZLIB
#ifdef LODEPNG_COMPILE_DECODER
unsigned lodepng_inflate(unsigned char** out, size_t* outsize, const unsigned char* in, size_t insize, const LodePNGDecompressSettings* settings);
unsigned lodepng_zlib_decompress(unsigned char** out, size_t* outsize, const unsigned char* in, size_t insize, const LodePNGDecompressSettings* settings);
#endif
#ifdef LODEPNG_COMPILE_ENCODER
unsigned lodepng_zlib_compress(unsigned char** out, size_t* outsize, const unsigned char* in, size_t insize, const LodePNGCompressSettings* settings);
unsigned lodepng_huffman_code_lengths(unsigned* lengths, const unsigned* frequencies, size_t numcodes, unsigned maxbitlen);
unsigned lodepng_deflate(unsigned char** out, size_t* outsize, const unsigned char* in, size_t insize, const LodePNGCompressSettings* settings);
#endif
#endif
#ifdef LODEPNG_COMPILE_DISK
unsigned lodepng_load_file(unsigned char** out, size_t* outsize, const char* filename);
unsigned lodepng_save_file(const unsigned char* buffer, size_t buffersize, const char* filename);
#endif
#ifdef LODEPNG_COMPILE_CPP
namespace lodepng
{
#ifdef LODEPNG_COMPILE_PNG
	class State : public LodePNGState
	{
	public:
		State();
		State(const State& other);
		virtual ~State();
		State& operator=(const State& other);
	};
#ifdef LODEPNG_COMPILE_DECODER
	unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h, State& state, const unsigned char* in, size_t insize);
	unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h, State& state, const std::vector<unsigned char>& in);
#endif
#ifdef LODEPNG_COMPILE_ENCODER
	unsigned encode(std::vector<unsigned char>& out, const unsigned char* in, unsigned w, unsigned h, State& state);
	unsigned encode(std::vector<unsigned char>& out, const std::vector<unsigned char>& in, unsigned w, unsigned h, State& state);
#endif
#ifdef LODEPNG_COMPILE_DISK
	unsigned load_file(std::vector<unsigned char>& buffer, const std::string& filename);
	unsigned save_file(const std::vector<unsigned char>& buffer, const std::string& filename);
#endif
#endif
#ifdef LODEPNG_COMPILE_ZLIB
#ifdef LODEPNG_COMPILE_DECODER
	unsigned decompress(std::vector<unsigned char>& out, const unsigned char* in, size_t insize, const LodePNGDecompressSettings& settings = lodepng_default_decompress_settings);
	unsigned decompress(std::vector<unsigned char>& out, const std::vector<unsigned char>& in, const LodePNGDecompressSettings& settings = lodepng_default_decompress_settings);
#endif
#ifdef LODEPNG_COMPILE_ENCODER
	unsigned compress(std::vector<unsigned char>& out, const unsigned char* in, size_t insize, const LodePNGCompressSettings& settings = lodepng_default_compress_settings);
	unsigned compress(std::vector<unsigned char>& out, const std::vector<unsigned char>& in, const LodePNGCompressSettings& settings = lodepng_default_compress_settings);
#endif
#endif
}
#endif
#endif
#include <stdio.h>
#include <stdlib.h>
#ifdef LODEPNG_COMPILE_CPP
#include <fstream>
#endif
#if defined(_MSC_VER) && (_MSC_VER >= 1310)
#pragma warning( disable : 4244 )
#pragma warning( disable : 4996 )
#endif
const char* LODEPNG_VERSION_STRING = "20151208";
#ifdef LODEPNG_COMPILE_ALLOCATORS
static void* lodepng_malloc(size_t size) { return malloc(size); }
static void* lodepng_realloc(void* ptr, size_t new_size) { return realloc(ptr, new_size); }
static void lodepng_free(void* ptr) { free(ptr); }
#else
void* lodepng_malloc(size_t size);
void* lodepng_realloc(void* ptr, size_t new_size);
void lodepng_free(void* ptr);
#endif
#define CERROR_BREAK(errorvar,code) \
{\
  errorvar = code;\
  break;\
}
#define ERROR_BREAK(code) CERROR_BREAK(error, code)
#define CERROR_RETURN_ERROR(errorvar,code) \
{\
  errorvar = code;\
  return code;\
}
#define CERROR_TRY_RETURN(call) \
{\
  unsigned error = call;\
  if(error) return error;\
}
#define CERROR_RETURN(errorvar,code) \
{\
  errorvar = code;\
  return;\
}
#ifdef LODEPNG_COMPILE_ZLIB
typedef struct uivector
{
	unsigned* data;
	size_t size;
	size_t allocsize;
} uivector;
static void uivector_cleanup(void* p)
{
	((uivector*)p)->size = ((uivector*)p)->allocsize = 0;
	lodepng_free(((uivector*)p)->data);
	((uivector*)p)->data = NULL;
}
static unsigned uivector_reserve(uivector* p, size_t allocsize)
{
	if (allocsize > p->allocsize)
	{
		size_t newsize = (allocsize > p->allocsize * 2) ? allocsize : (allocsize * 3 / 2);
		void* data = lodepng_realloc(p->data, newsize);
		if (data)
		{
			p->allocsize = newsize;
			p->data = (unsigned*)data;
		}
		else return 0;
	}
	return 1;
}
static unsigned uivector_resize(uivector* p, size_t size)
{
	if (!uivector_reserve(p, size * sizeof(unsigned))) return 0;
	p->size = size;
	return 1;
}
static unsigned uivector_resizev(uivector* p, size_t size, unsigned value)
{
	size_t oldsize = p->size, i;
	if (!uivector_resize(p, size)) return 0;
	for (i = oldsize; i < size; ++i) p->data[i] = value;
	return 1;
}
static void uivector_init(uivector* p)
{
	p->data = NULL;
	p->size = p->allocsize = 0;
}
#ifdef LODEPNG_COMPILE_ENCODER
static unsigned uivector_push_back(uivector* p, unsigned c)
{
	if (!uivector_resize(p, p->size + 1)) return 0;
	p->data[p->size - 1] = c;
	return 1;
}
#endif
#endif
typedef struct ucvector
{
	unsigned char* data;
	size_t size;
	size_t allocsize;
} ucvector;
static unsigned ucvector_reserve(ucvector* p, size_t allocsize)
{
	if (allocsize > p->allocsize)
	{
		size_t newsize = (allocsize > p->allocsize * 2) ? allocsize : (allocsize * 3 / 2);
		void* data = lodepng_realloc(p->data, newsize);
		if (data)
		{
			p->allocsize = newsize;
			p->data = (unsigned char*)data;
		}
		else return 0;
	}
	return 1;
}
static unsigned ucvector_resize(ucvector* p, size_t size)
{
	if (!ucvector_reserve(p, size * sizeof(unsigned char))) return 0;
	p->size = size;
	return 1;
}
#ifdef LODEPNG_COMPILE_PNG
static void ucvector_cleanup(void* p)
{
	((ucvector*)p)->size = ((ucvector*)p)->allocsize = 0;
	lodepng_free(((ucvector*)p)->data);
	((ucvector*)p)->data = NULL;
}
static void ucvector_init(ucvector* p)
{
	p->data = NULL;
	p->size = p->allocsize = 0;
}
#ifdef LODEPNG_COMPILE_DECODER
static unsigned ucvector_resizev(ucvector* p, size_t size, unsigned char value)
{
	size_t oldsize = p->size, i;
	if (!ucvector_resize(p, size)) return 0;
	for (i = oldsize; i < size; ++i) p->data[i] = value;
	return 1;
}
#endif
#endif
#ifdef LODEPNG_COMPILE_ZLIB
static void ucvector_init_buffer(ucvector* p, unsigned char* buffer, size_t size)
{
	p->data = buffer;
	p->allocsize = p->size = size;
}
#endif
#if (defined(LODEPNG_COMPILE_PNG) && defined(LODEPNG_COMPILE_ANCILLARY_CHUNKS)) || defined(LODEPNG_COMPILE_ENCODER)
static unsigned ucvector_push_back(ucvector* p, unsigned char c)
{
	if (!ucvector_resize(p, p->size + 1)) return 0;
	p->data[p->size - 1] = c;
	return 1;
}
#endif
#ifdef LODEPNG_COMPILE_PNG
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
static unsigned string_resize(char** out, size_t size)
{
	char* data = (char*)lodepng_realloc(*out, size + 1);
	if (data)
	{
		data[size] = 0;
		*out = data;
	}
	return data != 0;
}
static void string_init(char** out)
{
	*out = NULL;
	string_resize(out, 0);
}
static void string_cleanup(char** out)
{
	lodepng_free(*out);
	*out = NULL;
}
static void string_set(char** out, const char* in)
{
	size_t insize = strlen(in), i;
	if (string_resize(out, insize))
	{
		for (i = 0; i != insize; ++i)
		{
			(*out)[i] = in[i];
		}
	}
}
#endif
#endif
unsigned lodepng_read32bitInt(const unsigned char* buffer)
{
	return (unsigned)((buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3]);
}
#if defined(LODEPNG_COMPILE_PNG) || defined(LODEPNG_COMPILE_ENCODER)
static void lodepng_set32bitInt(unsigned char* buffer, unsigned value)
{
	buffer[0] = (unsigned char)((value >> 24) & 0xff);
	buffer[1] = (unsigned char)((value >> 16) & 0xff);
	buffer[2] = (unsigned char)((value >> 8) & 0xff);
	buffer[3] = (unsigned char)((value)& 0xff);
}
#endif
#ifdef LODEPNG_COMPILE_ENCODER
static void lodepng_add32bitInt(ucvector* buffer, unsigned value)
{
	ucvector_resize(buffer, buffer->size + 4);
	lodepng_set32bitInt(&buffer->data[buffer->size - 4], value);
}
#endif
#ifdef LODEPNG_COMPILE_DISK
unsigned lodepng_load_file(unsigned char** out, size_t* outsize, const char* filename)
{
	FILE* file;
	long size;
	*out = 0;
	*outsize = 0;
	file = fopen(filename, "rb");
	if (!file) return 78;
	fseek(file, 0, SEEK_END);
	size = ftell(file);
	rewind(file);
	*outsize = 0;
	*out = (unsigned char*)lodepng_malloc((size_t)size);
	if (size && (*out)) (*outsize) = fread(*out, 1, (size_t)size, file);
	fclose(file);
	if (!(*out) && size) return 83;
	return 0;
}
unsigned lodepng_save_file(const unsigned char* buffer, size_t buffersize, const char* filename)
{
	FILE* file;
	file = fopen(filename, "wb");
	if (!file) return 79;
	fwrite((char*)buffer, 1, buffersize, file);
	fclose(file);
	return 0;
}
#endif
#ifdef LODEPNG_COMPILE_ZLIB
#ifdef LODEPNG_COMPILE_ENCODER
#define addBitToStream(bitpointer,bitstream,bit) \
{\
                               \
  if(((*bitpointer) & 7) == 0) ucvector_push_back(bitstream, (unsigned char)0);\
                                                                                   \
  (bitstream->data[bitstream->size - 1]) |= (bit << ((*bitpointer) & 0x7));\
  ++(*bitpointer);\
}
static void addBitsToStream(size_t* bitpointer, ucvector* bitstream, unsigned value, size_t nbits)
{
	size_t i;
	for (i = 0; i != nbits; ++i) addBitToStream(bitpointer, bitstream, (unsigned char)((value >> i) & 1));
}
static void addBitsToStreamReversed(size_t* bitpointer, ucvector* bitstream, unsigned value, size_t nbits)
{
	size_t i;
	for (i = 0; i != nbits; ++i) addBitToStream(bitpointer, bitstream, (unsigned char)((value >> (nbits - 1 - i)) & 1));
}
#endif
#ifdef LODEPNG_COMPILE_DECODER
#define READBIT(bitpointer,bitstream) ((bitstream[bitpointer >> 3] >> (bitpointer & 0x7)) & (unsigned char)1)
static unsigned char readBitFromStream(size_t* bitpointer, const unsigned char* bitstream)
{
	unsigned char result = (unsigned char)(READBIT(*bitpointer, bitstream));
	++(*bitpointer);
	return result;
}
static unsigned readBitsFromStream(size_t* bitpointer, const unsigned char* bitstream, size_t nbits)
{
	unsigned result = 0, i;
	for (i = 0; i != nbits; ++i)
	{
		result += ((unsigned)READBIT(*bitpointer, bitstream)) << i;
		++(*bitpointer);
	}
	return result;
}
#endif
#define FIRST_LENGTH_CODE_INDEX 257
#define LAST_LENGTH_CODE_INDEX 285
#define NUM_DEFLATE_CODE_SYMBOLS 288
#define NUM_DISTANCE_SYMBOLS 32
#define NUM_CODE_LENGTH_CODES 19
static const unsigned LENGTHBASE[29] = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
static const unsigned LENGTHEXTRA[29] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
static const unsigned DISTANCEBASE[30] = { 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 };
static const unsigned DISTANCEEXTRA[30] = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };
static const unsigned CLCL_ORDER[NUM_CODE_LENGTH_CODES] = { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
typedef struct HuffmanTree
{
	unsigned* tree2d;
	unsigned* tree1d;
	unsigned* lengths;
	unsigned maxbitlen;
	unsigned numcodes;
} HuffmanTree;
static void HuffmanTree_init(HuffmanTree* tree)
{
	tree->tree2d = 0;
	tree->tree1d = 0;
	tree->lengths = 0;
}
static void HuffmanTree_cleanup(HuffmanTree* tree)
{
	lodepng_free(tree->tree2d);
	lodepng_free(tree->tree1d);
	lodepng_free(tree->lengths);
}
static unsigned HuffmanTree_make2DTree(HuffmanTree* tree)
{
	unsigned nodefilled = 0;
	unsigned treepos = 0;
	unsigned n, i;
	tree->tree2d = (unsigned*)lodepng_malloc(tree->numcodes * 2 * sizeof(unsigned));
	if (!tree->tree2d) return 83;
	for (n = 0; n < tree->numcodes * 2; ++n)
	{
		tree->tree2d[n] = 32767;
	}
	for (n = 0; n < tree->numcodes; ++n)
	{
		for (i = 0; i != tree->lengths[n]; ++i)
		{
			unsigned char bit = (unsigned char)((tree->tree1d[n] >> (tree->lengths[n] - i - 1)) & 1);
			if (treepos > 2147483647 || treepos + 2 > tree->numcodes) return 55;
			if (tree->tree2d[2 * treepos + bit] == 32767)
			{
				if (i + 1 == tree->lengths[n])
				{
					tree->tree2d[2 * treepos + bit] = n;
					treepos = 0;
				}
				else
				{
					++nodefilled;
					tree->tree2d[2 * treepos + bit] = nodefilled + tree->numcodes;
					treepos = nodefilled;
				}
			}
			else treepos = tree->tree2d[2 * treepos + bit] - tree->numcodes;
		}
	}
	for (n = 0; n < tree->numcodes * 2; ++n)
	{
		if (tree->tree2d[n] == 32767) tree->tree2d[n] = 0;
	}
	return 0;
}
static unsigned HuffmanTree_makeFromLengths2(HuffmanTree* tree)
{
	uivector blcount;
	uivector nextcode;
	unsigned error = 0;
	unsigned bits, n;
	uivector_init(&blcount);
	uivector_init(&nextcode);
	tree->tree1d = (unsigned*)lodepng_malloc(tree->numcodes * sizeof(unsigned));
	if (!tree->tree1d) error = 83;
	if (!uivector_resizev(&blcount, tree->maxbitlen + 1, 0)
		|| !uivector_resizev(&nextcode, tree->maxbitlen + 1, 0))
		error = 83;
	if (!error)
	{
		for (bits = 0; bits != tree->numcodes; ++bits) ++blcount.data[tree->lengths[bits]];
		for (bits = 1; bits <= tree->maxbitlen; ++bits)
		{
			nextcode.data[bits] = (nextcode.data[bits - 1] + blcount.data[bits - 1]) << 1;
		}
		for (n = 0; n != tree->numcodes; ++n)
		{
			if (tree->lengths[n] != 0) tree->tree1d[n] = nextcode.data[tree->lengths[n]]++;
		}
	}
	uivector_cleanup(&blcount);
	uivector_cleanup(&nextcode);
	if (!error) return HuffmanTree_make2DTree(tree);
	else return error;
}
static unsigned HuffmanTree_makeFromLengths(HuffmanTree* tree, const unsigned* bitlen, size_t numcodes, unsigned maxbitlen)
{
	unsigned i;
	tree->lengths = (unsigned*)lodepng_malloc(numcodes * sizeof(unsigned));
	if (!tree->lengths) return 83;
	for (i = 0; i != numcodes; ++i) tree->lengths[i] = bitlen[i];
	tree->numcodes = (unsigned)numcodes;
	tree->maxbitlen = maxbitlen;
	return HuffmanTree_makeFromLengths2(tree);
}
#ifdef LODEPNG_COMPILE_ENCODER
typedef struct BPMNode
{
	int weight;
	unsigned index;
	struct BPMNode* tail;
	int in_use;
} BPMNode;
typedef struct BPMLists
{
	unsigned memsize;
	BPMNode* memory;
	unsigned numfree;
	unsigned nextfree;
	BPMNode** freelist;
	unsigned listsize;
	BPMNode** chains0;
	BPMNode** chains1;
} BPMLists;
static BPMNode* bpmnode_create(BPMLists* lists, int weight, unsigned index, BPMNode* tail)
{
	unsigned i;
	BPMNode* result;
	if (lists->nextfree >= lists->numfree)
	{
		for (i = 0; i != lists->memsize; ++i) lists->memory[i].in_use = 0;
		for (i = 0; i != lists->listsize; ++i)
		{
			BPMNode* node;
			for (node = lists->chains0[i]; node != 0; node = node->tail) node->in_use = 1;
			for (node = lists->chains1[i]; node != 0; node = node->tail) node->in_use = 1;
		}
		lists->numfree = 0;
		for (i = 0; i != lists->memsize; ++i)
		{
			if (!lists->memory[i].in_use) lists->freelist[lists->numfree++] = &lists->memory[i];
		}
		lists->nextfree = 0;
	}
	result = lists->freelist[lists->nextfree++];
	result->weight = weight;
	result->index = index;
	result->tail = tail;
	return result;
}
static int bpmnode_compare(const void* a, const void* b)
{
	int wa = ((const BPMNode*)a)->weight;
	int wb = ((const BPMNode*)b)->weight;
	if (wa < wb) return -1;
	if (wa > wb) return 1;
	return ((const BPMNode*)a)->index < ((const BPMNode*)b)->index ? 1 : -1;
}
static void boundaryPM(BPMLists* lists, BPMNode* leaves, size_t numpresent, int c, int num)
{
	unsigned lastindex = lists->chains1[c]->index;
	if (c == 0)
	{
		if (lastindex >= numpresent) return;
		lists->chains0[c] = lists->chains1[c];
		lists->chains1[c] = bpmnode_create(lists, leaves[lastindex].weight, lastindex + 1, 0);
	}
	else
	{
		int sum = lists->chains0[c - 1]->weight + lists->chains1[c - 1]->weight;
		lists->chains0[c] = lists->chains1[c];
		if (lastindex < numpresent && sum > leaves[lastindex].weight)
		{
			lists->chains1[c] = bpmnode_create(lists, leaves[lastindex].weight, lastindex + 1, lists->chains1[c]->tail);
			return;
		}
		lists->chains1[c] = bpmnode_create(lists, sum, lastindex, lists->chains1[c - 1]);
		if (num + 1 < (int)(2 * numpresent - 2))
		{
			boundaryPM(lists, leaves, numpresent, c - 1, num);
			boundaryPM(lists, leaves, numpresent, c - 1, num);
		}
	}
}
unsigned lodepng_huffman_code_lengths(unsigned* lengths, const unsigned* frequencies, size_t numcodes, unsigned maxbitlen)
{
	unsigned error = 0;
	unsigned i;
	size_t numpresent = 0;
	BPMNode* leaves;
	if (numcodes == 0) return 80;
	if ((1u << maxbitlen) < numcodes) return 80;
	leaves = (BPMNode*)lodepng_malloc(numcodes * sizeof(*leaves));
	if (!leaves) return 83;
	for (i = 0; i != numcodes; ++i)
	{
		if (frequencies[i] > 0)
		{
			leaves[numpresent].weight = (int)frequencies[i];
			leaves[numpresent].index = i;
			++numpresent;
		}
	}
	for (i = 0; i != numcodes; ++i) lengths[i] = 0;
	if (numpresent == 0)
	{
		lengths[0] = lengths[1] = 1;
	}
	else if (numpresent == 1)
	{
		lengths[leaves[0].index] = 1;
		lengths[leaves[0].index == 0 ? 1 : 0] = 1;
	}
	else
	{
		BPMLists lists;
		BPMNode* node;
		qsort(leaves, numpresent, sizeof(BPMNode), bpmnode_compare);
		lists.listsize = maxbitlen;
		lists.memsize = 2 * maxbitlen * (maxbitlen + 1);
		lists.nextfree = 0;
		lists.numfree = lists.memsize;
		lists.memory = (BPMNode*)lodepng_malloc(lists.memsize * sizeof(*lists.memory));
		lists.freelist = (BPMNode**)lodepng_malloc(lists.memsize * sizeof(BPMNode*));
		lists.chains0 = (BPMNode**)lodepng_malloc(lists.listsize * sizeof(BPMNode*));
		lists.chains1 = (BPMNode**)lodepng_malloc(lists.listsize * sizeof(BPMNode*));
		if (!lists.memory || !lists.freelist || !lists.chains0 || !lists.chains1) error = 83;
		if (!error)
		{
			for (i = 0; i != lists.memsize; ++i) lists.freelist[i] = &lists.memory[i];
			bpmnode_create(&lists, leaves[0].weight, 1, 0);
			bpmnode_create(&lists, leaves[1].weight, 2, 0);
			for (i = 0; i != lists.listsize; ++i)
			{
				lists.chains0[i] = &lists.memory[0];
				lists.chains1[i] = &lists.memory[1];
			}
			for (i = 2; i != 2 * numpresent - 2; ++i) boundaryPM(&lists, leaves, numpresent, (int)maxbitlen - 1, (int)i);
			for (node = lists.chains1[maxbitlen - 1]; node; node = node->tail)
			{
				for (i = 0; i != node->index; ++i) ++lengths[leaves[i].index];
			}
		}
		lodepng_free(lists.memory);
		lodepng_free(lists.freelist);
		lodepng_free(lists.chains0);
		lodepng_free(lists.chains1);
	}
	lodepng_free(leaves);
	return error;
}
static unsigned HuffmanTree_makeFromFrequencies(HuffmanTree* tree, const unsigned* frequencies, size_t mincodes, size_t numcodes, unsigned maxbitlen)
{
	unsigned error = 0;
	while (!frequencies[numcodes - 1] && numcodes > mincodes) --numcodes;
	tree->maxbitlen = maxbitlen;
	tree->numcodes = (unsigned)numcodes;
	tree->lengths = (unsigned*)lodepng_realloc(tree->lengths, numcodes * sizeof(unsigned));
	if (!tree->lengths) return 83;
	memset(tree->lengths, 0, numcodes * sizeof(unsigned));
	error = lodepng_huffman_code_lengths(tree->lengths, frequencies, numcodes, maxbitlen);
	if (!error) error = HuffmanTree_makeFromLengths2(tree);
	return error;
}
static unsigned HuffmanTree_getCode(const HuffmanTree* tree, unsigned index)
{
	return tree->tree1d[index];
}
static unsigned HuffmanTree_getLength(const HuffmanTree* tree, unsigned index)
{
	return tree->lengths[index];
}
#endif
static unsigned generateFixedLitLenTree(HuffmanTree* tree)
{
	unsigned i, error = 0;
	unsigned* bitlen = (unsigned*)lodepng_malloc(NUM_DEFLATE_CODE_SYMBOLS * sizeof(unsigned));
	if (!bitlen) return 83;
	for (i = 0; i <= 143; ++i) bitlen[i] = 8;
	for (i = 144; i <= 255; ++i) bitlen[i] = 9;
	for (i = 256; i <= 279; ++i) bitlen[i] = 7;
	for (i = 280; i <= 287; ++i) bitlen[i] = 8;
	error = HuffmanTree_makeFromLengths(tree, bitlen, NUM_DEFLATE_CODE_SYMBOLS, 15);
	lodepng_free(bitlen);
	return error;
}
static unsigned generateFixedDistanceTree(HuffmanTree* tree)
{
	unsigned i, error = 0;
	unsigned* bitlen = (unsigned*)lodepng_malloc(NUM_DISTANCE_SYMBOLS * sizeof(unsigned));
	if (!bitlen) return 83;
	for (i = 0; i != NUM_DISTANCE_SYMBOLS; ++i) bitlen[i] = 5;
	error = HuffmanTree_makeFromLengths(tree, bitlen, NUM_DISTANCE_SYMBOLS, 15);
	lodepng_free(bitlen);
	return error;
}
#ifdef LODEPNG_COMPILE_DECODER
static unsigned huffmanDecodeSymbol(const unsigned char* in, size_t* bp, const HuffmanTree* codetree, size_t inbitlength)
{
	unsigned treepos = 0, ct;
	for (;;)
	{
		if (*bp >= inbitlength) return (unsigned)(-1);
		ct = codetree->tree2d[(treepos << 1) + READBIT(*bp, in)];
		++(*bp);
		if (ct < codetree->numcodes) return ct;
		else treepos = ct - codetree->numcodes;
		if (treepos >= codetree->numcodes) return (unsigned)(-1);
	}
}
#endif
#ifdef LODEPNG_COMPILE_DECODER
static void getTreeInflateFixed(HuffmanTree* tree_ll, HuffmanTree* tree_d)
{
	generateFixedLitLenTree(tree_ll);
	generateFixedDistanceTree(tree_d);
}
static unsigned getTreeInflateDynamic(HuffmanTree* tree_ll, HuffmanTree* tree_d, const unsigned char* in, size_t* bp, size_t inlength)
{
	unsigned error = 0;
	unsigned n, HLIT, HDIST, HCLEN, i;
	size_t inbitlength = inlength * 8;
	unsigned* bitlen_ll = 0;
	unsigned* bitlen_d = 0;
	unsigned* bitlen_cl = 0;
	HuffmanTree tree_cl;
	if ((*bp) + 14 > (inlength << 3)) return 49;
	HLIT = readBitsFromStream(bp, in, 5) + 257;
	HDIST = readBitsFromStream(bp, in, 5) + 1;
	HCLEN = readBitsFromStream(bp, in, 4) + 4;
	if ((*bp) + HCLEN * 3 > (inlength << 3)) return 50;
	HuffmanTree_init(&tree_cl);
	while (!error)
	{
		bitlen_cl = (unsigned*)lodepng_malloc(NUM_CODE_LENGTH_CODES * sizeof(unsigned));
		if (!bitlen_cl) ERROR_BREAK(83);
		for (i = 0; i != NUM_CODE_LENGTH_CODES; ++i)
		{
			if (i < HCLEN) bitlen_cl[CLCL_ORDER[i]] = readBitsFromStream(bp, in, 3);
			else bitlen_cl[CLCL_ORDER[i]] = 0;
		}
		error = HuffmanTree_makeFromLengths(&tree_cl, bitlen_cl, NUM_CODE_LENGTH_CODES, 7);
		if (error) break;
		bitlen_ll = (unsigned*)lodepng_malloc(NUM_DEFLATE_CODE_SYMBOLS * sizeof(unsigned));
		bitlen_d = (unsigned*)lodepng_malloc(NUM_DISTANCE_SYMBOLS * sizeof(unsigned));
		if (!bitlen_ll || !bitlen_d) ERROR_BREAK(83);
		for (i = 0; i != NUM_DEFLATE_CODE_SYMBOLS; ++i) bitlen_ll[i] = 0;
		for (i = 0; i != NUM_DISTANCE_SYMBOLS; ++i) bitlen_d[i] = 0;
		i = 0;
		while (i < HLIT + HDIST)
		{
			unsigned code = huffmanDecodeSymbol(in, bp, &tree_cl, inbitlength);
			if (code <= 15)
			{
				if (i < HLIT) bitlen_ll[i] = code;
				else bitlen_d[i - HLIT] = code;
				++i;
			}
			else if (code == 16)
			{
				unsigned replength = 3;
				unsigned value;
				if (i == 0) ERROR_BREAK(54);
				if ((*bp + 2) > inbitlength) ERROR_BREAK(50);
				replength += readBitsFromStream(bp, in, 2);
				if (i < HLIT + 1) value = bitlen_ll[i - 1];
				else value = bitlen_d[i - HLIT - 1];
				for (n = 0; n < replength; ++n)
				{
					if (i >= HLIT + HDIST) ERROR_BREAK(13);
					if (i < HLIT) bitlen_ll[i] = value;
					else bitlen_d[i - HLIT] = value;
					++i;
				}
			}
			else if (code == 17)
			{
				unsigned replength = 3;
				if ((*bp + 3) > inbitlength) ERROR_BREAK(50);
				replength += readBitsFromStream(bp, in, 3);
				for (n = 0; n < replength; ++n)
				{
					if (i >= HLIT + HDIST) ERROR_BREAK(14);

					if (i < HLIT) bitlen_ll[i] = 0;
					else bitlen_d[i - HLIT] = 0;
					++i;
				}
			}
			else if (code == 18)
			{
				unsigned replength = 11;
				if ((*bp + 7) > inbitlength) ERROR_BREAK(50);
				replength += readBitsFromStream(bp, in, 7);
				for (n = 0; n < replength; ++n)
				{
					if (i >= HLIT + HDIST) ERROR_BREAK(15);

					if (i < HLIT) bitlen_ll[i] = 0;
					else bitlen_d[i - HLIT] = 0;
					++i;
				}
			}
			else
			{
				if (code == (unsigned)(-1))
				{
					error = (*bp) > inbitlength ? 10 : 11;
				}
				else error = 16;
				break;
			}
		}
		if (error) break;
		if (bitlen_ll[256] == 0) ERROR_BREAK(64);
		error = HuffmanTree_makeFromLengths(tree_ll, bitlen_ll, NUM_DEFLATE_CODE_SYMBOLS, 15);
		if (error) break;
		error = HuffmanTree_makeFromLengths(tree_d, bitlen_d, NUM_DISTANCE_SYMBOLS, 15);
		break;
	}
	lodepng_free(bitlen_cl);
	lodepng_free(bitlen_ll);
	lodepng_free(bitlen_d);
	HuffmanTree_cleanup(&tree_cl);
	return error;
}
static unsigned inflateHuffmanBlock(ucvector* out, const unsigned char* in, size_t* bp, size_t* pos, size_t inlength, unsigned btype)
{
	unsigned error = 0;
	HuffmanTree tree_ll;
	HuffmanTree tree_d;
	size_t inbitlength = inlength * 8;
	HuffmanTree_init(&tree_ll);
	HuffmanTree_init(&tree_d);
	if (btype == 1) getTreeInflateFixed(&tree_ll, &tree_d);
	else if (btype == 2) error = getTreeInflateDynamic(&tree_ll, &tree_d, in, bp, inlength);
	while (!error)
	{
		unsigned code_ll = huffmanDecodeSymbol(in, bp, &tree_ll, inbitlength);
		if (code_ll <= 255)
		{
			if (!ucvector_resize(out, (*pos) + 1)) ERROR_BREAK(83);
			out->data[*pos] = (unsigned char)code_ll;
			++(*pos);
		}
		else if (code_ll >= FIRST_LENGTH_CODE_INDEX && code_ll <= LAST_LENGTH_CODE_INDEX)
		{
			unsigned code_d, distance;
			unsigned numextrabits_l, numextrabits_d;
			size_t start, forward, backward, length;
			length = LENGTHBASE[code_ll - FIRST_LENGTH_CODE_INDEX];
			numextrabits_l = LENGTHEXTRA[code_ll - FIRST_LENGTH_CODE_INDEX];
			if ((*bp + numextrabits_l) > inbitlength) ERROR_BREAK(51);
			length += readBitsFromStream(bp, in, numextrabits_l);
			code_d = huffmanDecodeSymbol(in, bp, &tree_d, inbitlength);
			if (code_d > 29)
			{
				if (code_ll == (unsigned)(-1))
				{
					error = (*bp) > inlength * 8 ? 10 : 11;
				}
				else error = 18;
				break;
			}
			distance = DISTANCEBASE[code_d];
			numextrabits_d = DISTANCEEXTRA[code_d];
			if ((*bp + numextrabits_d) > inbitlength) ERROR_BREAK(51);
			distance += readBitsFromStream(bp, in, numextrabits_d);
			start = (*pos);
			if (distance > start) ERROR_BREAK(52);
			backward = start - distance;
			if (!ucvector_resize(out, (*pos) + length)) ERROR_BREAK(83);
			if (distance < length) {
				for (forward = 0; forward < length; ++forward)
				{
					out->data[(*pos)++] = out->data[backward++];
				}
			}
			else {
				memcpy(out->data + *pos, out->data + backward, length);
				*pos += length;
			}
		}
		else if (code_ll == 256)
		{
			break;
		}
		else
		{
			error = ((*bp) > inlength * 8) ? 10 : 11;
			break;
		}
	}
	HuffmanTree_cleanup(&tree_ll);
	HuffmanTree_cleanup(&tree_d);
	return error;
}

static unsigned inflateNoCompression(ucvector* out, const unsigned char* in, size_t* bp, size_t* pos, size_t inlength)
{
	size_t p;
	unsigned LEN, NLEN, n, error = 0;
	while (((*bp) & 0x7) != 0) ++(*bp);
	p = (*bp) / 8;
	if (p + 4 >= inlength) return 52;
	LEN = in[p] + 256u * in[p + 1]; p += 2;
	NLEN = in[p] + 256u * in[p + 1]; p += 2;
	if (LEN + NLEN != 65535) return 21;
	if (!ucvector_resize(out, (*pos) + LEN)) return 83;
	if (p + LEN > inlength) return 23;
	for (n = 0; n < LEN; ++n) out->data[(*pos)++] = in[p++];
	(*bp) = p * 8;
	return error;
}

static unsigned lodepng_inflatev(ucvector* out, const unsigned char* in, size_t insize, const LodePNGDecompressSettings* settings)
{
	size_t bp = 0;
	unsigned BFINAL = 0;
	size_t pos = 0;
	unsigned error = 0;
	(void)settings;
	while (!BFINAL)
	{
		unsigned BTYPE;
		if (bp + 2 >= insize * 8) return 52;
		BFINAL = readBitFromStream(&bp, in);
		BTYPE = 1u * readBitFromStream(&bp, in);
		BTYPE += 2u * readBitFromStream(&bp, in);
		if (BTYPE == 3) return 20;
		else if (BTYPE == 0) error = inflateNoCompression(out, in, &bp, &pos, insize);
		else error = inflateHuffmanBlock(out, in, &bp, &pos, insize, BTYPE);
		if (error) return error;
	}
	return error;
}
unsigned lodepng_inflate(unsigned char** out, size_t* outsize, const unsigned char* in, size_t insize, const LodePNGDecompressSettings* settings)
{
	unsigned error;
	ucvector v;
	ucvector_init_buffer(&v, *out, *outsize);
	error = lodepng_inflatev(&v, in, insize, settings);
	*out = v.data;
	*outsize = v.size;
	return error;
}
static unsigned inflate(unsigned char** out, size_t* outsize,
	const unsigned char* in, size_t insize,
	const LodePNGDecompressSettings* settings)
{
	if (settings->custom_inflate)
	{
		return settings->custom_inflate(out, outsize, in, insize, settings);
	}
	else
	{
		return lodepng_inflate(out, outsize, in, insize, settings);
	}
}
#endif
#ifdef LODEPNG_COMPILE_ENCODER
static const size_t MAX_SUPPORTED_DEFLATE_LENGTH = 258;
static void addHuffmanSymbol(size_t* bp, ucvector* compressed, unsigned code, unsigned bitlen)
{
	addBitsToStreamReversed(bp, compressed, code, bitlen);
}
static size_t searchCodeIndex(const unsigned* array, size_t array_size, size_t value)
{
	size_t left = 1;
	size_t right = array_size - 1;
	while (left <= right)
	{
		size_t mid = (left + right) / 2;
		if (array[mid] <= value) left = mid + 1;
		else if (array[mid - 1] > value) right = mid - 1;
		else return mid - 1;
	}
	return array_size - 1;
}
static void addLengthDistance(uivector* values, size_t length, size_t distance)
{
	unsigned length_code = (unsigned)searchCodeIndex(LENGTHBASE, 29, length);
	unsigned extra_length = (unsigned)(length - LENGTHBASE[length_code]);
	unsigned dist_code = (unsigned)searchCodeIndex(DISTANCEBASE, 30, distance);
	unsigned extra_distance = (unsigned)(distance - DISTANCEBASE[dist_code]);
	uivector_push_back(values, length_code + FIRST_LENGTH_CODE_INDEX);
	uivector_push_back(values, extra_length);
	uivector_push_back(values, dist_code);
	uivector_push_back(values, extra_distance);
}
static const unsigned HASH_NUM_VALUES = 65536;
static const unsigned HASH_BIT_MASK = 65535;
typedef struct Hash
{
	int* head;
	unsigned short* chain;
	int* val;
	int* headz;
	unsigned short* chainz;
	unsigned short* zeros;
} Hash;
static unsigned hash_init(Hash* hash, unsigned windowsize)
{
	unsigned i;
	hash->head = (int*)lodepng_malloc(sizeof(int) * HASH_NUM_VALUES);
	hash->val = (int*)lodepng_malloc(sizeof(int) * windowsize);
	hash->chain = (unsigned short*)lodepng_malloc(sizeof(unsigned short) * windowsize);
	hash->zeros = (unsigned short*)lodepng_malloc(sizeof(unsigned short) * windowsize);
	hash->headz = (int*)lodepng_malloc(sizeof(int) * (MAX_SUPPORTED_DEFLATE_LENGTH + 1));
	hash->chainz = (unsigned short*)lodepng_malloc(sizeof(unsigned short) * windowsize);
	if (!hash->head || !hash->chain || !hash->val || !hash->headz || !hash->chainz || !hash->zeros)
	{
		return 83;
	}
	for (i = 0; i != HASH_NUM_VALUES; ++i) hash->head[i] = -1;
	for (i = 0; i != windowsize; ++i) hash->val[i] = -1;
	for (i = 0; i != windowsize; ++i) hash->chain[i] = i;
	for (i = 0; i <= MAX_SUPPORTED_DEFLATE_LENGTH; ++i) hash->headz[i] = -1;
	for (i = 0; i != windowsize; ++i) hash->chainz[i] = i;
	return 0;
}
static void hash_cleanup(Hash* hash)
{
	lodepng_free(hash->head);
	lodepng_free(hash->val);
	lodepng_free(hash->chain);
	lodepng_free(hash->zeros);
	lodepng_free(hash->headz);
	lodepng_free(hash->chainz);
}
static unsigned getHash(const unsigned char* data, size_t size, size_t pos)
{
	unsigned result = 0;
	if (pos + 2 < size)
	{
		result ^= (unsigned)(data[pos + 0] << 0u);
		result ^= (unsigned)(data[pos + 1] << 4u);
		result ^= (unsigned)(data[pos + 2] << 8u);
	}
	else {
		size_t amount, i;
		if (pos >= size) return 0;
		amount = size - pos;
		for (i = 0; i != amount; ++i) result ^= (unsigned)(data[pos + i] << (i * 8u));
	}
	return result & HASH_BIT_MASK;
}
static unsigned countZeros(const unsigned char* data, size_t size, size_t pos)
{
	const unsigned char* start = data + pos;
	const unsigned char* end = start + MAX_SUPPORTED_DEFLATE_LENGTH;
	if (end > data + size) end = data + size;
	data = start;
	while (data != end && *data == 0) ++data;
	return (unsigned)(data - start);
}
static void updateHashChain(Hash* hash, size_t wpos, unsigned hashval, unsigned short numzeros)
{
	hash->val[wpos] = (int)hashval;
	if (hash->head[hashval] != -1) hash->chain[wpos] = hash->head[hashval];
	hash->head[hashval] = wpos;
	hash->zeros[wpos] = numzeros;
	if (hash->headz[numzeros] != -1) hash->chainz[wpos] = hash->headz[numzeros];
	hash->headz[numzeros] = wpos;
}
static unsigned encodeLZ77(uivector* out, Hash* hash,
	const unsigned char* in, size_t inpos, size_t insize, unsigned windowsize,
	unsigned minmatch, unsigned nicematch, unsigned lazymatching)
{
	size_t pos;
	unsigned i, error = 0;

	unsigned maxchainlength = windowsize >= 8192 ? windowsize : windowsize / 8;
	unsigned maxlazymatch = windowsize >= 8192 ? MAX_SUPPORTED_DEFLATE_LENGTH : 64;

	unsigned usezeros = 1;
	unsigned numzeros = 0;

	unsigned offset;
	unsigned length;
	unsigned lazy = 0;
	unsigned lazylength = 0, lazyoffset = 0;
	unsigned hashval;
	unsigned current_offset, current_length;
	unsigned prev_offset;
	const unsigned char *lastptr, *foreptr, *backptr;
	unsigned hashpos;

	if (windowsize == 0 || windowsize > 32768) return 60;
	if ((windowsize & (windowsize - 1)) != 0) return 90;

	if (nicematch > MAX_SUPPORTED_DEFLATE_LENGTH) nicematch = MAX_SUPPORTED_DEFLATE_LENGTH;

	for (pos = inpos; pos < insize; ++pos)
	{
		size_t wpos = pos & (windowsize - 1);
		unsigned chainlength = 0;

		hashval = getHash(in, insize, pos);

		if (usezeros && hashval == 0)
		{
			if (numzeros == 0) numzeros = countZeros(in, insize, pos);
			else if (pos + numzeros > insize || in[pos + numzeros - 1] != 0) --numzeros;
		}
		else
		{
			numzeros = 0;
		}

		updateHashChain(hash, wpos, hashval, numzeros);


		length = 0;
		offset = 0;

		hashpos = hash->chain[wpos];

		lastptr = &in[insize < pos + MAX_SUPPORTED_DEFLATE_LENGTH ? insize : pos + MAX_SUPPORTED_DEFLATE_LENGTH];


		prev_offset = 0;
		for (;;)
		{
			if (chainlength++ >= maxchainlength) break;
			current_offset = hashpos <= wpos ? wpos - hashpos : wpos - hashpos + windowsize;

			if (current_offset < prev_offset) break;
			prev_offset = current_offset;
			if (current_offset > 0)
			{

				foreptr = &in[pos];
				backptr = &in[pos - current_offset];


				if (numzeros >= 3)
				{
					unsigned skip = hash->zeros[hashpos];
					if (skip > numzeros) skip = numzeros;
					backptr += skip;
					foreptr += skip;
				}

				while (foreptr != lastptr && *backptr == *foreptr)
				{
					++backptr;
					++foreptr;
				}
				current_length = (unsigned)(foreptr - &in[pos]);

				if (current_length > length)
				{
					length = current_length;
					offset = current_offset;


					if (current_length >= nicematch) break;
				}
			}

			if (hashpos == hash->chain[hashpos]) break;

			if (numzeros >= 3 && length > numzeros)
			{
				hashpos = hash->chainz[hashpos];
				if (hash->zeros[hashpos] != numzeros) break;
			}
			else
			{
				hashpos = hash->chain[hashpos];

				if (hash->val[hashpos] != (int)hashval) break;
			}
		}

		if (lazymatching)
		{
			if (!lazy && length >= 3 && length <= maxlazymatch && length < MAX_SUPPORTED_DEFLATE_LENGTH)
			{
				lazy = 1;
				lazylength = length;
				lazyoffset = offset;
				continue;
			}
			if (lazy)
			{
				lazy = 0;
				if (pos == 0) ERROR_BREAK(81);
				if (length > lazylength + 1)
				{

					if (!uivector_push_back(out, in[pos - 1])) ERROR_BREAK(83);
				}
				else
				{
					length = lazylength;
					offset = lazyoffset;
					hash->head[hashval] = -1;
					hash->headz[numzeros] = -1;
					--pos;
				}
			}
		}
		if (length >= 3 && offset > windowsize) ERROR_BREAK(86);


		if (length < 3)
		{
			if (!uivector_push_back(out, in[pos])) ERROR_BREAK(83);
		}
		else if (length < minmatch || (length == 3 && offset > 4096))
		{


			if (!uivector_push_back(out, in[pos])) ERROR_BREAK(83);
		}
		else
		{
			addLengthDistance(out, length, offset);
			for (i = 1; i < length; ++i)
			{
				++pos;
				wpos = pos & (windowsize - 1);
				hashval = getHash(in, insize, pos);
				if (usezeros && hashval == 0)
				{
					if (numzeros == 0) numzeros = countZeros(in, insize, pos);
					else if (pos + numzeros > insize || in[pos + numzeros - 1] != 0) --numzeros;
				}
				else
				{
					numzeros = 0;
				}
				updateHashChain(hash, wpos, hashval, numzeros);
			}
		}
	}

	return error;
}



static unsigned deflateNoCompression(ucvector* out, const unsigned char* data, size_t datasize)
{



	size_t i, j, numdeflateblocks = (datasize + 65534) / 65535;
	unsigned datapos = 0;
	for (i = 0; i != numdeflateblocks; ++i)
	{
		unsigned BFINAL, BTYPE, LEN, NLEN;
		unsigned char firstbyte;

		BFINAL = (i == numdeflateblocks - 1);
		BTYPE = 0;

		firstbyte = (unsigned char)(BFINAL + ((BTYPE & 1) << 1) + ((BTYPE & 2) << 1));
		ucvector_push_back(out, firstbyte);

		LEN = 65535;
		if (datasize - datapos < 65535) LEN = (unsigned)datasize - datapos;
		NLEN = 65535 - LEN;

		ucvector_push_back(out, (unsigned char)(LEN % 256));
		ucvector_push_back(out, (unsigned char)(LEN / 256));
		ucvector_push_back(out, (unsigned char)(NLEN % 256));
		ucvector_push_back(out, (unsigned char)(NLEN / 256));


		for (j = 0; j < 65535 && datapos < datasize; ++j)
		{
			ucvector_push_back(out, data[datapos++]);
		}
	}

	return 0;
}






static void writeLZ77data(size_t* bp, ucvector* out, const uivector* lz77_encoded,
	const HuffmanTree* tree_ll, const HuffmanTree* tree_d)
{
	size_t i = 0;
	for (i = 0; i != lz77_encoded->size; ++i)
	{
		unsigned val = lz77_encoded->data[i];
		addHuffmanSymbol(bp, out, HuffmanTree_getCode(tree_ll, val), HuffmanTree_getLength(tree_ll, val));
		if (val > 256)
		{
			unsigned length_index = val - FIRST_LENGTH_CODE_INDEX;
			unsigned n_length_extra_bits = LENGTHEXTRA[length_index];
			unsigned length_extra_bits = lz77_encoded->data[++i];

			unsigned distance_code = lz77_encoded->data[++i];

			unsigned distance_index = distance_code;
			unsigned n_distance_extra_bits = DISTANCEEXTRA[distance_index];
			unsigned distance_extra_bits = lz77_encoded->data[++i];

			addBitsToStream(bp, out, length_extra_bits, n_length_extra_bits);
			addHuffmanSymbol(bp, out, HuffmanTree_getCode(tree_d, distance_code),
				HuffmanTree_getLength(tree_d, distance_code));
			addBitsToStream(bp, out, distance_extra_bits, n_distance_extra_bits);
		}
	}
}


static unsigned deflateDynamic(ucvector* out, size_t* bp, Hash* hash,
	const unsigned char* data, size_t datapos, size_t dataend,
	const LodePNGCompressSettings* settings, unsigned final)
{
	unsigned error = 0;
		uivector lz77_encoded;
	HuffmanTree tree_ll;
	HuffmanTree tree_d;
	HuffmanTree tree_cl;
	uivector frequencies_ll;
	uivector frequencies_d;
	uivector frequencies_cl;
	uivector bitlen_lld;
	uivector bitlen_lld_e;



	uivector bitlen_cl;
	size_t datasize = dataend - datapos;
		unsigned BFINAL = final;
	size_t numcodes_ll, numcodes_d, i;
	unsigned HLIT, HDIST, HCLEN;

	uivector_init(&lz77_encoded);
	HuffmanTree_init(&tree_ll);
	HuffmanTree_init(&tree_d);
	HuffmanTree_init(&tree_cl);
	uivector_init(&frequencies_ll);
	uivector_init(&frequencies_d);
	uivector_init(&frequencies_cl);
	uivector_init(&bitlen_lld);
	uivector_init(&bitlen_lld_e);
	uivector_init(&bitlen_cl);



	while (!error)
	{
		if (settings->use_lz77)
		{
			error = encodeLZ77(&lz77_encoded, hash, data, datapos, dataend, settings->windowsize,
				settings->minmatch, settings->nicematch, settings->lazymatching);
			if (error) break;
		}
		else
		{
			if (!uivector_resize(&lz77_encoded, datasize)) ERROR_BREAK(83);
			for (i = datapos; i < dataend; ++i) lz77_encoded.data[i] = data[i];
		}

		if (!uivector_resizev(&frequencies_ll, 286, 0)) ERROR_BREAK(83);
		if (!uivector_resizev(&frequencies_d, 30, 0)) ERROR_BREAK(83);


		for (i = 0; i != lz77_encoded.size; ++i)
		{
			unsigned symbol = lz77_encoded.data[i];
			++frequencies_ll.data[symbol];
			if (symbol > 256)
			{
				unsigned dist = lz77_encoded.data[i + 2];
				++frequencies_d.data[dist];
				i += 3;
			}
		}
		frequencies_ll.data[256] = 1;


		error = HuffmanTree_makeFromFrequencies(&tree_ll, frequencies_ll.data, 257, frequencies_ll.size, 15);
		if (error) break;

		error = HuffmanTree_makeFromFrequencies(&tree_d, frequencies_d.data, 2, frequencies_d.size, 15);
		if (error) break;

		numcodes_ll = tree_ll.numcodes; if (numcodes_ll > 286) numcodes_ll = 286;
		numcodes_d = tree_d.numcodes; if (numcodes_d > 30) numcodes_d = 30;

		for (i = 0; i != numcodes_ll; ++i) uivector_push_back(&bitlen_lld, HuffmanTree_getLength(&tree_ll, (unsigned)i));
		for (i = 0; i != numcodes_d; ++i) uivector_push_back(&bitlen_lld, HuffmanTree_getLength(&tree_d, (unsigned)i));



		for (i = 0; i != (unsigned)bitlen_lld.size; ++i)
		{
			unsigned j = 0;
			while (i + j + 1 < (unsigned)bitlen_lld.size && bitlen_lld.data[i + j + 1] == bitlen_lld.data[i]) ++j;

			if (bitlen_lld.data[i] == 0 && j >= 2)
			{
				++j;
				if (j <= 10)
				{
					uivector_push_back(&bitlen_lld_e, 17);
					uivector_push_back(&bitlen_lld_e, j - 3);
				}
				else
				{
					if (j > 138) j = 138;
					uivector_push_back(&bitlen_lld_e, 18);
					uivector_push_back(&bitlen_lld_e, j - 11);
				}
				i += (j - 1);
			}
			else if (j >= 3)
			{
				size_t k;
				unsigned num = j / 6, rest = j % 6;
				uivector_push_back(&bitlen_lld_e, bitlen_lld.data[i]);
				for (k = 0; k < num; ++k)
				{
					uivector_push_back(&bitlen_lld_e, 16);
					uivector_push_back(&bitlen_lld_e, 6 - 3);
				}
				if (rest >= 3)
				{
					uivector_push_back(&bitlen_lld_e, 16);
					uivector_push_back(&bitlen_lld_e, rest - 3);
				}
				else j -= rest;
				i += j;
			}
			else
			{
				uivector_push_back(&bitlen_lld_e, bitlen_lld.data[i]);
			}
		}



		if (!uivector_resizev(&frequencies_cl, NUM_CODE_LENGTH_CODES, 0)) ERROR_BREAK(83);
		for (i = 0; i != bitlen_lld_e.size; ++i)
		{
			++frequencies_cl.data[bitlen_lld_e.data[i]];


			if (bitlen_lld_e.data[i] >= 16) ++i;
		}

		error = HuffmanTree_makeFromFrequencies(&tree_cl, frequencies_cl.data,
			frequencies_cl.size, frequencies_cl.size, 7);
		if (error) break;

		if (!uivector_resize(&bitlen_cl, tree_cl.numcodes)) ERROR_BREAK(83);
		for (i = 0; i != tree_cl.numcodes; ++i)
		{

			bitlen_cl.data[i] = HuffmanTree_getLength(&tree_cl, CLCL_ORDER[i]);
		}
		while (bitlen_cl.data[bitlen_cl.size - 1] == 0 && bitlen_cl.size > 4)
		{

			if (!uivector_resize(&bitlen_cl, bitlen_cl.size - 1)) ERROR_BREAK(83);
		}
		if (error) break;
			addBitToStream(bp, out, BFINAL);
		addBitToStream(bp, out, 0);
		addBitToStream(bp, out, 1);


		HLIT = (unsigned)(numcodes_ll - 257);
		HDIST = (unsigned)(numcodes_d - 1);
		HCLEN = (unsigned)bitlen_cl.size - 4;

		while (!bitlen_cl.data[HCLEN + 4 - 1] && HCLEN > 0) --HCLEN;
		addBitsToStream(bp, out, HLIT, 5);
		addBitsToStream(bp, out, HDIST, 5);
		addBitsToStream(bp, out, HCLEN, 4);


		for (i = 0; i != HCLEN + 4; ++i) addBitsToStream(bp, out, bitlen_cl.data[i], 3);


		for (i = 0; i != bitlen_lld_e.size; ++i)
		{
			addHuffmanSymbol(bp, out, HuffmanTree_getCode(&tree_cl, bitlen_lld_e.data[i]),
				HuffmanTree_getLength(&tree_cl, bitlen_lld_e.data[i]));

			if (bitlen_lld_e.data[i] == 16) addBitsToStream(bp, out, bitlen_lld_e.data[++i], 2);
			else if (bitlen_lld_e.data[i] == 17) addBitsToStream(bp, out, bitlen_lld_e.data[++i], 3);
			else if (bitlen_lld_e.data[i] == 18) addBitsToStream(bp, out, bitlen_lld_e.data[++i], 7);
		}


		writeLZ77data(bp, out, &lz77_encoded, &tree_ll, &tree_d);

		if (HuffmanTree_getLength(&tree_ll, 256) == 0) ERROR_BREAK(64);


		addHuffmanSymbol(bp, out, HuffmanTree_getCode(&tree_ll, 256), HuffmanTree_getLength(&tree_ll, 256));

		break;
	}


	uivector_cleanup(&lz77_encoded);
	HuffmanTree_cleanup(&tree_ll);
	HuffmanTree_cleanup(&tree_d);
	HuffmanTree_cleanup(&tree_cl);
	uivector_cleanup(&frequencies_ll);
	uivector_cleanup(&frequencies_d);
	uivector_cleanup(&frequencies_cl);
	uivector_cleanup(&bitlen_lld_e);
	uivector_cleanup(&bitlen_lld);
	uivector_cleanup(&bitlen_cl);

	return error;
}

static unsigned deflateFixed(ucvector* out, size_t* bp, Hash* hash,
	const unsigned char* data,
	size_t datapos, size_t dataend,
	const LodePNGCompressSettings* settings, unsigned final)
{
	HuffmanTree tree_ll;
	HuffmanTree tree_d;

	unsigned BFINAL = final;
	unsigned error = 0;
	size_t i;

	HuffmanTree_init(&tree_ll);
	HuffmanTree_init(&tree_d);

	generateFixedLitLenTree(&tree_ll);
	generateFixedDistanceTree(&tree_d);

	addBitToStream(bp, out, BFINAL);
	addBitToStream(bp, out, 1);
	addBitToStream(bp, out, 0);

	if (settings->use_lz77)
	{
		uivector lz77_encoded;
		uivector_init(&lz77_encoded);
		error = encodeLZ77(&lz77_encoded, hash, data, datapos, dataend, settings->windowsize,
			settings->minmatch, settings->nicematch, settings->lazymatching);
		if (!error) writeLZ77data(bp, out, &lz77_encoded, &tree_ll, &tree_d);
		uivector_cleanup(&lz77_encoded);
	}
	else
	{
		for (i = datapos; i < dataend; ++i)
		{
			addHuffmanSymbol(bp, out, HuffmanTree_getCode(&tree_ll, data[i]), HuffmanTree_getLength(&tree_ll, data[i]));
		}
	}

	if (!error) addHuffmanSymbol(bp, out, HuffmanTree_getCode(&tree_ll, 256), HuffmanTree_getLength(&tree_ll, 256));


	HuffmanTree_cleanup(&tree_ll);
	HuffmanTree_cleanup(&tree_d);

	return error;
}

static unsigned lodepng_deflatev(ucvector* out, const unsigned char* in, size_t insize,
	const LodePNGCompressSettings* settings)
{
	unsigned error = 0;
	size_t i, blocksize, numdeflateblocks;
	size_t bp = 0;
	Hash hash;

	if (settings->btype > 2) return 61;
	else if (settings->btype == 0) return deflateNoCompression(out, in, insize);
	else if (settings->btype == 1) blocksize = insize;
	else
	{

		blocksize = insize / 8 + 8;
		if (blocksize < 65536) blocksize = 65536;
		if (blocksize > 262144) blocksize = 262144;
	}

	numdeflateblocks = (insize + blocksize - 1) / blocksize;
	if (numdeflateblocks == 0) numdeflateblocks = 1;

	error = hash_init(&hash, settings->windowsize);
	if (error) return error;

	for (i = 0; i != numdeflateblocks && !error; ++i)
	{
		unsigned final = (i == numdeflateblocks - 1);
		size_t start = i * blocksize;
		size_t end = start + blocksize;
		if (end > insize) end = insize;

		if (settings->btype == 1) error = deflateFixed(out, &bp, &hash, in, start, end, settings, final);
		else if (settings->btype == 2) error = deflateDynamic(out, &bp, &hash, in, start, end, settings, final);
	}

	hash_cleanup(&hash);

	return error;
}

unsigned lodepng_deflate(unsigned char** out, size_t* outsize,
	const unsigned char* in, size_t insize,
	const LodePNGCompressSettings* settings)
{
	unsigned error;
	ucvector v;
	ucvector_init_buffer(&v, *out, *outsize);
	error = lodepng_deflatev(&v, in, insize, settings);
	*out = v.data;
	*outsize = v.size;
	return error;
}

static unsigned deflate(unsigned char** out, size_t* outsize,
	const unsigned char* in, size_t insize,
	const LodePNGCompressSettings* settings)
{
	if (settings->custom_deflate)
	{
		return settings->custom_deflate(out, outsize, in, insize, settings);
	}
	else
	{
		return lodepng_deflate(out, outsize, in, insize, settings);
	}
}

#endif





static unsigned update_adler32(unsigned adler, const unsigned char* data, unsigned len)
{
	unsigned s1 = adler & 0xffff;
	unsigned s2 = (adler >> 16) & 0xffff;

	while (len > 0)
	{

		unsigned amount = len > 5550 ? 5550 : len;
		len -= amount;
		while (amount > 0)
		{
			s1 += (*data++);
			s2 += s1;
			--amount;
		}
		s1 %= 65521;
		s2 %= 65521;
	}

	return (s2 << 16) | s1;
}


static unsigned adler32(const unsigned char* data, unsigned len)
{
	return update_adler32(1L, data, len);
}





#ifdef LODEPNG_COMPILE_DECODER

unsigned lodepng_zlib_decompress(unsigned char** out, size_t* outsize, const unsigned char* in,
	size_t insize, const LodePNGDecompressSettings* settings)
{
	unsigned error = 0;
	unsigned CM, CINFO, FDICT;

	if (insize < 2) return 53;

	if ((in[0] * 256 + in[1]) % 31 != 0)
	{

		return 24;
	}

	CM = in[0] & 15;
	CINFO = (in[0] >> 4) & 15;

	FDICT = (in[1] >> 5) & 1;


	if (CM != 8 || CINFO > 7)
	{

		return 25;
	}
	if (FDICT != 0)
	{


		return 26;
	}

	error = inflate(out, outsize, in + 2, insize - 2, settings);
	if (error) return error;

	if (!settings->ignore_adler32)
	{
		unsigned ADLER32 = lodepng_read32bitInt(&in[insize - 4]);
		unsigned checksum = adler32(*out, (unsigned)(*outsize));
		if (checksum != ADLER32) return 58;
	}

	return 0;
}

static unsigned zlib_decompress(unsigned char** out, size_t* outsize, const unsigned char* in,
	size_t insize, const LodePNGDecompressSettings* settings)
{
	if (settings->custom_zlib)
	{
		return settings->custom_zlib(out, outsize, in, insize, settings);
	}
	else
	{
		return lodepng_zlib_decompress(out, outsize, in, insize, settings);
	}
}

#endif

#ifdef LODEPNG_COMPILE_ENCODER

unsigned lodepng_zlib_compress(unsigned char** out, size_t* outsize, const unsigned char* in,
	size_t insize, const LodePNGCompressSettings* settings)
{


	ucvector outv;
	size_t i;
	unsigned error;
	unsigned char* deflatedata = 0;
	size_t deflatesize = 0;


	unsigned CMF = 120;
	unsigned FLEVEL = 0;
	unsigned FDICT = 0;
	unsigned CMFFLG = 256 * CMF + FDICT * 32 + FLEVEL * 64;
	unsigned FCHECK = 31 - CMFFLG % 31;
	CMFFLG += FCHECK;


	ucvector_init_buffer(&outv, *out, *outsize);

	ucvector_push_back(&outv, (unsigned char)(CMFFLG / 256));
	ucvector_push_back(&outv, (unsigned char)(CMFFLG % 256));

	error = deflate(&deflatedata, &deflatesize, in, insize, settings);

	if (!error)
	{
		unsigned ADLER32 = adler32(in, (unsigned)insize);
		for (i = 0; i != deflatesize; ++i) ucvector_push_back(&outv, deflatedata[i]);
		lodepng_free(deflatedata);
		lodepng_add32bitInt(&outv, ADLER32);
	}

	*out = outv.data;
	*outsize = outv.size;

	return error;
}


static unsigned zlib_compress(unsigned char** out, size_t* outsize, const unsigned char* in,
	size_t insize, const LodePNGCompressSettings* settings)
{
	if (settings->custom_zlib)
	{
		return settings->custom_zlib(out, outsize, in, insize, settings);
	}
	else
	{
		return lodepng_zlib_compress(out, outsize, in, insize, settings);
	}
}

#endif

#else

#ifdef LODEPNG_COMPILE_DECODER
static unsigned zlib_decompress(unsigned char** out, size_t* outsize, const unsigned char* in,
	size_t insize, const LodePNGDecompressSettings* settings)
{
	if (!settings->custom_zlib) return 87;
	return settings->custom_zlib(out, outsize, in, insize, settings);
}
#endif
#ifdef LODEPNG_COMPILE_ENCODER
static unsigned zlib_compress(unsigned char** out, size_t* outsize, const unsigned char* in,
	size_t insize, const LodePNGCompressSettings* settings)
{
	if (!settings->custom_zlib) return 87;
	return settings->custom_zlib(out, outsize, in, insize, settings);
}
#endif

#endif



#ifdef LODEPNG_COMPILE_ENCODER


#define DEFAULT_WINDOWSIZE 2048

void lodepng_compress_settings_init(LodePNGCompressSettings* settings)
{

	settings->btype = 2;
	settings->use_lz77 = 1;
	settings->windowsize = DEFAULT_WINDOWSIZE;
	settings->minmatch = 3;
	settings->nicematch = 128;
	settings->lazymatching = 1;

	settings->custom_zlib = 0;
	settings->custom_deflate = 0;
	settings->custom_context = 0;
}

const LodePNGCompressSettings lodepng_default_compress_settings = { 2, 1, DEFAULT_WINDOWSIZE, 3, 128, 1, 0, 0, 0 };


#endif

#ifdef LODEPNG_COMPILE_DECODER

void lodepng_decompress_settings_init(LodePNGDecompressSettings* settings)
{
	settings->ignore_adler32 = 0;

	settings->custom_zlib = 0;
	settings->custom_inflate = 0;
	settings->custom_context = 0;
}

const LodePNGDecompressSettings lodepng_default_decompress_settings = { 0, 0, 0, 0 };

#endif







#ifdef LODEPNG_COMPILE_PNG






#ifndef LODEPNG_NO_COMPILE_CRC

static unsigned lodepng_crc32_table[256] = {
	0u, 1996959894u, 3993919788u, 2567524794u, 124634137u, 1886057615u, 3915621685u, 2657392035u,
	249268274u, 2044508324u, 3772115230u, 2547177864u, 162941995u, 2125561021u, 3887607047u, 2428444049u,
	498536548u, 1789927666u, 4089016648u, 2227061214u, 450548861u, 1843258603u, 4107580753u, 2211677639u,
	325883990u, 1684777152u, 4251122042u, 2321926636u, 335633487u, 1661365465u, 4195302755u, 2366115317u,
	997073096u, 1281953886u, 3579855332u, 2724688242u, 1006888145u, 1258607687u, 3524101629u, 2768942443u,
	901097722u, 1119000684u, 3686517206u, 2898065728u, 853044451u, 1172266101u, 3705015759u, 2882616665u,
	651767980u, 1373503546u, 3369554304u, 3218104598u, 565507253u, 1454621731u, 3485111705u, 3099436303u,
	671266974u, 1594198024u, 3322730930u, 2970347812u, 795835527u, 1483230225u, 3244367275u, 3060149565u,
	1994146192u, 31158534u, 2563907772u, 4023717930u, 1907459465u, 112637215u, 2680153253u, 3904427059u,
	2013776290u, 251722036u, 2517215374u, 3775830040u, 2137656763u, 141376813u, 2439277719u, 3865271297u,
	1802195444u, 476864866u, 2238001368u, 4066508878u, 1812370925u, 453092731u, 2181625025u, 4111451223u,
	1706088902u, 314042704u, 2344532202u, 4240017532u, 1658658271u, 366619977u, 2362670323u, 4224994405u,
	1303535960u, 984961486u, 2747007092u, 3569037538u, 1256170817u, 1037604311u, 2765210733u, 3554079995u,
	1131014506u, 879679996u, 2909243462u, 3663771856u, 1141124467u, 855842277u, 2852801631u, 3708648649u,
	1342533948u, 654459306u, 3188396048u, 3373015174u, 1466479909u, 544179635u, 3110523913u, 3462522015u,
	1591671054u, 702138776u, 2966460450u, 3352799412u, 1504918807u, 783551873u, 3082640443u, 3233442989u,
	3988292384u, 2596254646u, 62317068u, 1957810842u, 3939845945u, 2647816111u, 81470997u, 1943803523u,
	3814918930u, 2489596804u, 225274430u, 2053790376u, 3826175755u, 2466906013u, 167816743u, 2097651377u,
	4027552580u, 2265490386u, 503444072u, 1762050814u, 4150417245u, 2154129355u, 426522225u, 1852507879u,
	4275313526u, 2312317920u, 282753626u, 1742555852u, 4189708143u, 2394877945u, 397917763u, 1622183637u,
	3604390888u, 2714866558u, 953729732u, 1340076626u, 3518719985u, 2797360999u, 1068828381u, 1219638859u,
	3624741850u, 2936675148u, 906185462u, 1090812512u, 3747672003u, 2825379669u, 829329135u, 1181335161u,
	3412177804u, 3160834842u, 628085408u, 1382605366u, 3423369109u, 3138078467u, 570562233u, 1426400815u,
	3317316542u, 2998733608u, 733239954u, 1555261956u, 3268935591u, 3050360625u, 752459403u, 1541320221u,
	2607071920u, 3965973030u, 1969922972u, 40735498u, 2617837225u, 3943577151u, 1913087877u, 83908371u,
	2512341634u, 3803740692u, 2075208622u, 213261112u, 2463272603u, 3855990285u, 2094854071u, 198958881u,
	2262029012u, 4057260610u, 1759359992u, 534414190u, 2176718541u, 4139329115u, 1873836001u, 414664567u,
	2282248934u, 4279200368u, 1711684554u, 285281116u, 2405801727u, 4167216745u, 1634467795u, 376229701u,
	2685067896u, 3608007406u, 1308918612u, 956543938u, 2808555105u, 3495958263u, 1231636301u, 1047427035u,
	2932959818u, 3654703836u, 1088359270u, 936918000u, 2847714899u, 3736837829u, 1202900863u, 817233897u,
	3183342108u, 3401237130u, 1404277552u, 615818150u, 3134207493u, 3453421203u, 1423857449u, 601450431u,
	3009837614u, 3294710456u, 1567103746u, 711928724u, 3020668471u, 3272380065u, 1510334235u, 755167117u
};


unsigned lodepng_crc32(const unsigned char* buf, size_t len)
{
	unsigned c = 0xffffffffL;
	size_t n;

	for (n = 0; n < len; ++n)
	{
		c = lodepng_crc32_table[(c ^ buf[n]) & 0xff] ^ (c >> 8);
	}
	return c ^ 0xffffffffL;
}
#endif





static unsigned char readBitFromReversedStream(size_t* bitpointer, const unsigned char* bitstream)
{
	unsigned char result = (unsigned char)((bitstream[(*bitpointer) >> 3] >> (7 - ((*bitpointer) & 0x7))) & 1);
	++(*bitpointer);
	return result;
}

static unsigned readBitsFromReversedStream(size_t* bitpointer, const unsigned char* bitstream, size_t nbits)
{
	unsigned result = 0;
	size_t i;
	for (i = nbits - 1; i < nbits; --i)
	{
		result += (unsigned)readBitFromReversedStream(bitpointer, bitstream) << i;
	}
	return result;
}

#ifdef LODEPNG_COMPILE_DECODER
static void setBitOfReversedStream0(size_t* bitpointer, unsigned char* bitstream, unsigned char bit)
{

	if (bit)
	{

		bitstream[(*bitpointer) >> 3] |= (bit << (7 - ((*bitpointer) & 0x7)));
	}
	++(*bitpointer);
}
#endif

static void setBitOfReversedStream(size_t* bitpointer, unsigned char* bitstream, unsigned char bit)
{

	if (bit == 0) bitstream[(*bitpointer) >> 3] &= (unsigned char)(~(1 << (7 - ((*bitpointer) & 0x7))));
	else bitstream[(*bitpointer) >> 3] |= (1 << (7 - ((*bitpointer) & 0x7)));
	++(*bitpointer);
}





unsigned lodepng_chunk_length(const unsigned char* chunk)
{
	return lodepng_read32bitInt(&chunk[0]);
}

void lodepng_chunk_type(char type[5], const unsigned char* chunk)
{
	unsigned i;
	for (i = 0; i != 4; ++i) type[i] = (char)chunk[4 + i];
	type[4] = 0;
}

unsigned char lodepng_chunk_type_equals(const unsigned char* chunk, const char* type)
{
	if (strlen(type) != 4) return 0;
	return (chunk[4] == type[0] && chunk[5] == type[1] && chunk[6] == type[2] && chunk[7] == type[3]);
}

unsigned char lodepng_chunk_ancillary(const unsigned char* chunk)
{
	return((chunk[4] & 32) != 0);
}

unsigned char lodepng_chunk_private(const unsigned char* chunk)
{
	return((chunk[6] & 32) != 0);
}

unsigned char lodepng_chunk_safetocopy(const unsigned char* chunk)
{
	return((chunk[7] & 32) != 0);
}

unsigned char* lodepng_chunk_data(unsigned char* chunk)
{
	return &chunk[8];
}

const unsigned char* lodepng_chunk_data_const(const unsigned char* chunk)
{
	return &chunk[8];
}

unsigned lodepng_chunk_check_crc(const unsigned char* chunk)
{
	unsigned length = lodepng_chunk_length(chunk);
	unsigned CRC = lodepng_read32bitInt(&chunk[length + 8]);

	unsigned checksum = lodepng_crc32(&chunk[4], length + 4);
	if (CRC != checksum) return 1;
	else return 0;
}

void lodepng_chunk_generate_crc(unsigned char* chunk)
{
	unsigned length = lodepng_chunk_length(chunk);
	unsigned CRC = lodepng_crc32(&chunk[4], length + 4);
	lodepng_set32bitInt(chunk + 8 + length, CRC);
}

unsigned char* lodepng_chunk_next(unsigned char* chunk)
{
	unsigned total_chunk_length = lodepng_chunk_length(chunk) + 12;
	return &chunk[total_chunk_length];
}

const unsigned char* lodepng_chunk_next_const(const unsigned char* chunk)
{
	unsigned total_chunk_length = lodepng_chunk_length(chunk) + 12;
	return &chunk[total_chunk_length];
}

unsigned lodepng_chunk_append(unsigned char** out, size_t* outlength, const unsigned char* chunk)
{
	unsigned i;
	unsigned total_chunk_length = lodepng_chunk_length(chunk) + 12;
	unsigned char *chunk_start, *new_buffer;
	size_t new_length = (*outlength) + total_chunk_length;
	if (new_length < total_chunk_length || new_length < (*outlength)) return 77;

	new_buffer = (unsigned char*)lodepng_realloc(*out, new_length);
	if (!new_buffer) return 83;
	(*out) = new_buffer;
	(*outlength) = new_length;
	chunk_start = &(*out)[new_length - total_chunk_length];

	for (i = 0; i != total_chunk_length; ++i) chunk_start[i] = chunk[i];

	return 0;
}

unsigned lodepng_chunk_create(unsigned char** out, size_t* outlength, unsigned length,
	const char* type, const unsigned char* data)
{
	unsigned i;
	unsigned char *chunk, *new_buffer;
	size_t new_length = (*outlength) + length + 12;
	if (new_length < length + 12 || new_length < (*outlength)) return 77;
	new_buffer = (unsigned char*)lodepng_realloc(*out, new_length);
	if (!new_buffer) return 83;
	(*out) = new_buffer;
	(*outlength) = new_length;
	chunk = &(*out)[(*outlength) - length - 12];


	lodepng_set32bitInt(chunk, (unsigned)length);


	chunk[4] = (unsigned char)type[0];
	chunk[5] = (unsigned char)type[1];
	chunk[6] = (unsigned char)type[2];
	chunk[7] = (unsigned char)type[3];


	for (i = 0; i != length; ++i) chunk[8 + i] = data[i];


	lodepng_chunk_generate_crc(chunk);

	return 0;
}






static unsigned checkColorValidity(LodePNGColorType colortype, unsigned bd)
{
	switch (colortype)
	{
	case 0: if (!(bd == 1 || bd == 2 || bd == 4 || bd == 8 || bd == 16)) return 37; break;
	case 2: if (!(bd == 8 || bd == 16)) return 37; break;
	case 3: if (!(bd == 1 || bd == 2 || bd == 4 || bd == 8)) return 37; break;
	case 4: if (!(bd == 8 || bd == 16)) return 37; break;
	case 6: if (!(bd == 8 || bd == 16)) return 37; break;
	default: return 31;
	}
	return 0;
}

static unsigned getNumColorChannels(LodePNGColorType colortype)
{
	switch (colortype)
	{
	case 0: return 1;
	case 2: return 3;
	case 3: return 1;
	case 4: return 2;
	case 6: return 4;
	}
	return 0;
}

static unsigned lodepng_get_bpp_lct(LodePNGColorType colortype, unsigned bitdepth)
{

	return getNumColorChannels(colortype) * bitdepth;
}



void lodepng_color_mode_init(LodePNGColorMode* info)
{
	info->key_defined = 0;
	info->key_r = info->key_g = info->key_b = 0;
	info->colortype = LCT_RGBA;
	info->bitdepth = 8;
	info->palette = 0;
	info->palettesize = 0;
}

void lodepng_color_mode_cleanup(LodePNGColorMode* info)
{
	lodepng_palette_clear(info);
}

unsigned lodepng_color_mode_copy(LodePNGColorMode* dest, const LodePNGColorMode* source)
{
	size_t i;
	lodepng_color_mode_cleanup(dest);
	*dest = *source;
	if (source->palette)
	{
		dest->palette = (unsigned char*)lodepng_malloc(1024);
		if (!dest->palette && source->palettesize) return 83;
		for (i = 0; i != source->palettesize * 4; ++i) dest->palette[i] = source->palette[i];
	}
	return 0;
}

static int lodepng_color_mode_equal(const LodePNGColorMode* a, const LodePNGColorMode* b)
{
	size_t i;
	if (a->colortype != b->colortype) return 0;
	if (a->bitdepth != b->bitdepth) return 0;
	if (a->key_defined != b->key_defined) return 0;
	if (a->key_defined)
	{
		if (a->key_r != b->key_r) return 0;
		if (a->key_g != b->key_g) return 0;
		if (a->key_b != b->key_b) return 0;
	}



	if (1) {
		if (a->palettesize != b->palettesize) return 0;
		for (i = 0; i != a->palettesize * 4; ++i)
		{
			if (a->palette[i] != b->palette[i]) return 0;
		}
	}
	return 1;
}

void lodepng_palette_clear(LodePNGColorMode* info)
{
	if (info->palette) lodepng_free(info->palette);
	info->palette = 0;
	info->palettesize = 0;
}

unsigned lodepng_palette_add(LodePNGColorMode* info,
	unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	unsigned char* data;


	if (!info->palette)
	{

		data = (unsigned char*)lodepng_realloc(info->palette, 1024);
		if (!data) return 83;
		else info->palette = data;
	}
	info->palette[4 * info->palettesize + 0] = r;
	info->palette[4 * info->palettesize + 1] = g;
	info->palette[4 * info->palettesize + 2] = b;
	info->palette[4 * info->palettesize + 3] = a;
	++info->palettesize;
	return 0;
}

unsigned lodepng_get_bpp(const LodePNGColorMode* info)
{

	return lodepng_get_bpp_lct(info->colortype, info->bitdepth);
}

unsigned lodepng_get_channels(const LodePNGColorMode* info)
{
	return getNumColorChannels(info->colortype);
}

unsigned lodepng_is_greyscale_type(const LodePNGColorMode* info)
{
	return info->colortype == LCT_GREY || info->colortype == LCT_GREY_ALPHA;
}

unsigned lodepng_is_alpha_type(const LodePNGColorMode* info)
{
	return (info->colortype & 4) != 0;
}

unsigned lodepng_is_palette_type(const LodePNGColorMode* info)
{
	return info->colortype == LCT_PALETTE;
}

unsigned lodepng_has_palette_alpha(const LodePNGColorMode* info)
{
	size_t i;
	for (i = 0; i != info->palettesize; ++i)
	{
		if (info->palette[i * 4 + 3] < 255) return 1;
	}
	return 0;
}

unsigned lodepng_can_have_alpha(const LodePNGColorMode* info)
{
	return info->key_defined
		|| lodepng_is_alpha_type(info)
		|| lodepng_has_palette_alpha(info);
}

size_t lodepng_get_raw_size(unsigned w, unsigned h, const LodePNGColorMode* color)
{
	return (w * h * lodepng_get_bpp(color) + 7) / 8;
}

size_t lodepng_get_raw_size_lct(unsigned w, unsigned h, LodePNGColorType colortype, unsigned bitdepth)
{
	return (w * h * lodepng_get_bpp_lct(colortype, bitdepth) + 7) / 8;
}


#ifdef LODEPNG_COMPILE_PNG
#ifdef LODEPNG_COMPILE_DECODER

static size_t lodepng_get_raw_size_idat(unsigned w, unsigned h, const LodePNGColorMode* color)
{
	return h * ((w * lodepng_get_bpp(color) + 7) / 8);
}
#endif
#endif

#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS

static void LodePNGUnknownChunks_init(LodePNGInfo* info)
{
	unsigned i;
	for (i = 0; i != 3; ++i) info->unknown_chunks_data[i] = 0;
	for (i = 0; i != 3; ++i) info->unknown_chunks_size[i] = 0;
}

static void LodePNGUnknownChunks_cleanup(LodePNGInfo* info)
{
	unsigned i;
	for (i = 0; i != 3; ++i) lodepng_free(info->unknown_chunks_data[i]);
}

static unsigned LodePNGUnknownChunks_copy(LodePNGInfo* dest, const LodePNGInfo* src)
{
	unsigned i;

	LodePNGUnknownChunks_cleanup(dest);

	for (i = 0; i != 3; ++i)
	{
		size_t j;
		dest->unknown_chunks_size[i] = src->unknown_chunks_size[i];
		dest->unknown_chunks_data[i] = (unsigned char*)lodepng_malloc(src->unknown_chunks_size[i]);
		if (!dest->unknown_chunks_data[i] && dest->unknown_chunks_size[i]) return 83;
		for (j = 0; j < src->unknown_chunks_size[i]; ++j)
		{
			dest->unknown_chunks_data[i][j] = src->unknown_chunks_data[i][j];
		}
	}

	return 0;
}



static void LodePNGText_init(LodePNGInfo* info)
{
	info->text_num = 0;
	info->text_keys = NULL;
	info->text_strings = NULL;
}

static void LodePNGText_cleanup(LodePNGInfo* info)
{
	size_t i;
	for (i = 0; i != info->text_num; ++i)
	{
		string_cleanup(&info->text_keys[i]);
		string_cleanup(&info->text_strings[i]);
	}
	lodepng_free(info->text_keys);
	lodepng_free(info->text_strings);
}

static unsigned LodePNGText_copy(LodePNGInfo* dest, const LodePNGInfo* source)
{
	size_t i = 0;
	dest->text_keys = 0;
	dest->text_strings = 0;
	dest->text_num = 0;
	for (i = 0; i != source->text_num; ++i)
	{
		CERROR_TRY_RETURN(lodepng_add_text(dest, source->text_keys[i], source->text_strings[i]));
	}
	return 0;
}

void lodepng_clear_text(LodePNGInfo* info)
{
	LodePNGText_cleanup(info);
}

unsigned lodepng_add_text(LodePNGInfo* info, const char* key, const char* str)
{
	char** new_keys = (char**)(lodepng_realloc(info->text_keys, sizeof(char*) * (info->text_num + 1)));
	char** new_strings = (char**)(lodepng_realloc(info->text_strings, sizeof(char*) * (info->text_num + 1)));
	if (!new_keys || !new_strings)
	{
		lodepng_free(new_keys);
		lodepng_free(new_strings);
		return 83;
	}

	++info->text_num;
	info->text_keys = new_keys;
	info->text_strings = new_strings;

	string_init(&info->text_keys[info->text_num - 1]);
	string_set(&info->text_keys[info->text_num - 1], key);

	string_init(&info->text_strings[info->text_num - 1]);
	string_set(&info->text_strings[info->text_num - 1], str);

	return 0;
}



static void LodePNGIText_init(LodePNGInfo* info)
{
	info->itext_num = 0;
	info->itext_keys = NULL;
	info->itext_langtags = NULL;
	info->itext_transkeys = NULL;
	info->itext_strings = NULL;
}

static void LodePNGIText_cleanup(LodePNGInfo* info)
{
	size_t i;
	for (i = 0; i != info->itext_num; ++i)
	{
		string_cleanup(&info->itext_keys[i]);
		string_cleanup(&info->itext_langtags[i]);
		string_cleanup(&info->itext_transkeys[i]);
		string_cleanup(&info->itext_strings[i]);
	}
	lodepng_free(info->itext_keys);
	lodepng_free(info->itext_langtags);
	lodepng_free(info->itext_transkeys);
	lodepng_free(info->itext_strings);
}

static unsigned LodePNGIText_copy(LodePNGInfo* dest, const LodePNGInfo* source)
{
	size_t i = 0;
	dest->itext_keys = 0;
	dest->itext_langtags = 0;
	dest->itext_transkeys = 0;
	dest->itext_strings = 0;
	dest->itext_num = 0;
	for (i = 0; i != source->itext_num; ++i)
	{
		CERROR_TRY_RETURN(lodepng_add_itext(dest, source->itext_keys[i], source->itext_langtags[i],
			source->itext_transkeys[i], source->itext_strings[i]));
	}
	return 0;
}

void lodepng_clear_itext(LodePNGInfo* info)
{
	LodePNGIText_cleanup(info);
}

unsigned lodepng_add_itext(LodePNGInfo* info, const char* key, const char* langtag,
	const char* transkey, const char* str)
{
	char** new_keys = (char**)(lodepng_realloc(info->itext_keys, sizeof(char*) * (info->itext_num + 1)));
	char** new_langtags = (char**)(lodepng_realloc(info->itext_langtags, sizeof(char*) * (info->itext_num + 1)));
	char** new_transkeys = (char**)(lodepng_realloc(info->itext_transkeys, sizeof(char*) * (info->itext_num + 1)));
	char** new_strings = (char**)(lodepng_realloc(info->itext_strings, sizeof(char*) * (info->itext_num + 1)));
	if (!new_keys || !new_langtags || !new_transkeys || !new_strings)
	{
		lodepng_free(new_keys);
		lodepng_free(new_langtags);
		lodepng_free(new_transkeys);
		lodepng_free(new_strings);
		return 83;
	}

	++info->itext_num;
	info->itext_keys = new_keys;
	info->itext_langtags = new_langtags;
	info->itext_transkeys = new_transkeys;
	info->itext_strings = new_strings;

	string_init(&info->itext_keys[info->itext_num - 1]);
	string_set(&info->itext_keys[info->itext_num - 1], key);

	string_init(&info->itext_langtags[info->itext_num - 1]);
	string_set(&info->itext_langtags[info->itext_num - 1], langtag);

	string_init(&info->itext_transkeys[info->itext_num - 1]);
	string_set(&info->itext_transkeys[info->itext_num - 1], transkey);

	string_init(&info->itext_strings[info->itext_num - 1]);
	string_set(&info->itext_strings[info->itext_num - 1], str);

	return 0;
}
#endif

void lodepng_info_init(LodePNGInfo* info)
{
	lodepng_color_mode_init(&info->color);
	info->interlace_method = 0;
	info->compression_method = 0;
	info->filter_method = 0;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
	info->background_defined = 0;
	info->background_r = info->background_g = info->background_b = 0;

	LodePNGText_init(info);
	LodePNGIText_init(info);

	info->time_defined = 0;
	info->phys_defined = 0;

	LodePNGUnknownChunks_init(info);
#endif
}

void lodepng_info_cleanup(LodePNGInfo* info)
{
	lodepng_color_mode_cleanup(&info->color);
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
	LodePNGText_cleanup(info);
	LodePNGIText_cleanup(info);

	LodePNGUnknownChunks_cleanup(info);
#endif
}

unsigned lodepng_info_copy(LodePNGInfo* dest, const LodePNGInfo* source)
{
	lodepng_info_cleanup(dest);
	*dest = *source;
	lodepng_color_mode_init(&dest->color);
	CERROR_TRY_RETURN(lodepng_color_mode_copy(&dest->color, &source->color));

#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
	CERROR_TRY_RETURN(LodePNGText_copy(dest, source));
	CERROR_TRY_RETURN(LodePNGIText_copy(dest, source));

	LodePNGUnknownChunks_init(dest);
	CERROR_TRY_RETURN(LodePNGUnknownChunks_copy(dest, source));
#endif
	return 0;
}

void lodepng_info_swap(LodePNGInfo* a, LodePNGInfo* b)
{
	LodePNGInfo temp = *a;
	*a = *b;
	*b = temp;
}




static void addColorBits(unsigned char* out, size_t index, unsigned bits, unsigned in)
{
	unsigned m = bits == 1 ? 7 : bits == 2 ? 3 : 1;

	unsigned p = index & m;
	in &= (1u << bits) - 1u;
	in = in << (bits * (m - p));
	if (p == 0) out[index * bits / 8] = in;
	else out[index * bits / 8] |= in;
}

typedef struct ColorTree ColorTree;







struct ColorTree
{
	ColorTree* children[16];
	int index;
};

static void color_tree_init(ColorTree* tree)
{
	int i;
	for (i = 0; i != 16; ++i) tree->children[i] = 0;
	tree->index = -1;
}

static void color_tree_cleanup(ColorTree* tree)
{
	int i;
	for (i = 0; i != 16; ++i)
	{
		if (tree->children[i])
		{
			color_tree_cleanup(tree->children[i]);
			lodepng_free(tree->children[i]);
		}
	}
}


static int color_tree_get(ColorTree* tree, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	int bit = 0;
	for (bit = 0; bit < 8; ++bit)
	{
		int i = 8 * ((r >> bit) & 1) + 4 * ((g >> bit) & 1) + 2 * ((b >> bit) & 1) + 1 * ((a >> bit) & 1);
		if (!tree->children[i]) return -1;
		else tree = tree->children[i];
	}
	return tree ? tree->index : -1;
}

#ifdef LODEPNG_COMPILE_ENCODER
static int color_tree_has(ColorTree* tree, unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	return color_tree_get(tree, r, g, b, a) >= 0;
}
#endif



static void color_tree_add(ColorTree* tree,
	unsigned char r, unsigned char g, unsigned char b, unsigned char a, unsigned index)
{
	int bit;
	for (bit = 0; bit < 8; ++bit)
	{
		int i = 8 * ((r >> bit) & 1) + 4 * ((g >> bit) & 1) + 2 * ((b >> bit) & 1) + 1 * ((a >> bit) & 1);
		if (!tree->children[i])
		{
			tree->children[i] = (ColorTree*)lodepng_malloc(sizeof(ColorTree));
			color_tree_init(tree->children[i]);
		}
		tree = tree->children[i];
	}
	tree->index = (int)index;
}


static unsigned rgba8ToPixel(unsigned char* out, size_t i,
	const LodePNGColorMode* mode, ColorTree* tree,
	unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
	if (mode->colortype == LCT_GREY)
	{
		unsigned char grey = r; ;
		if (mode->bitdepth == 8) out[i] = grey;
		else if (mode->bitdepth == 16) out[i * 2 + 0] = out[i * 2 + 1] = grey;
		else
		{

			grey = (grey >> (8 - mode->bitdepth)) & ((1 << mode->bitdepth) - 1);
			addColorBits(out, i, mode->bitdepth, grey);
		}
	}
	else if (mode->colortype == LCT_RGB)
	{
		if (mode->bitdepth == 8)
		{
			out[i * 3 + 0] = r;
			out[i * 3 + 1] = g;
			out[i * 3 + 2] = b;
		}
		else
		{
			out[i * 6 + 0] = out[i * 6 + 1] = r;
			out[i * 6 + 2] = out[i * 6 + 3] = g;
			out[i * 6 + 4] = out[i * 6 + 5] = b;
		}
	}
	else if (mode->colortype == LCT_PALETTE)
	{
		int index = color_tree_get(tree, r, g, b, a);
		if (index < 0) return 82;
		if (mode->bitdepth == 8) out[i] = index;
		else addColorBits(out, i, mode->bitdepth, (unsigned)index);
	}
	else if (mode->colortype == LCT_GREY_ALPHA)
	{
		unsigned char grey = r; ;
		if (mode->bitdepth == 8)
		{
			out[i * 2 + 0] = grey;
			out[i * 2 + 1] = a;
		}
		else if (mode->bitdepth == 16)
		{
			out[i * 4 + 0] = out[i * 4 + 1] = grey;
			out[i * 4 + 2] = out[i * 4 + 3] = a;
		}
	}
	else if (mode->colortype == LCT_RGBA)
	{
		if (mode->bitdepth == 8)
		{
			out[i * 4 + 0] = r;
			out[i * 4 + 1] = g;
			out[i * 4 + 2] = b;
			out[i * 4 + 3] = a;
		}
		else
		{
			out[i * 8 + 0] = out[i * 8 + 1] = r;
			out[i * 8 + 2] = out[i * 8 + 3] = g;
			out[i * 8 + 4] = out[i * 8 + 5] = b;
			out[i * 8 + 6] = out[i * 8 + 7] = a;
		}
	}

	return 0;
}


static void rgba16ToPixel(unsigned char* out, size_t i,
	const LodePNGColorMode* mode,
	unsigned short r, unsigned short g, unsigned short b, unsigned short a)
{
	if (mode->colortype == LCT_GREY)
	{
		unsigned short grey = r; ;
		out[i * 2 + 0] = (grey >> 8) & 255;
		out[i * 2 + 1] = grey & 255;
	}
	else if (mode->colortype == LCT_RGB)
	{
		out[i * 6 + 0] = (r >> 8) & 255;
		out[i * 6 + 1] = r & 255;
		out[i * 6 + 2] = (g >> 8) & 255;
		out[i * 6 + 3] = g & 255;
		out[i * 6 + 4] = (b >> 8) & 255;
		out[i * 6 + 5] = b & 255;
	}
	else if (mode->colortype == LCT_GREY_ALPHA)
	{
		unsigned short grey = r; ;
		out[i * 4 + 0] = (grey >> 8) & 255;
		out[i * 4 + 1] = grey & 255;
		out[i * 4 + 2] = (a >> 8) & 255;
		out[i * 4 + 3] = a & 255;
	}
	else if (mode->colortype == LCT_RGBA)
	{
		out[i * 8 + 0] = (r >> 8) & 255;
		out[i * 8 + 1] = r & 255;
		out[i * 8 + 2] = (g >> 8) & 255;
		out[i * 8 + 3] = g & 255;
		out[i * 8 + 4] = (b >> 8) & 255;
		out[i * 8 + 5] = b & 255;
		out[i * 8 + 6] = (a >> 8) & 255;
		out[i * 8 + 7] = a & 255;
	}
}


static void getPixelColorRGBA8(unsigned char* r, unsigned char* g,
	unsigned char* b, unsigned char* a,
	const unsigned char* in, size_t i,
	const LodePNGColorMode* mode)
{
	if (mode->colortype == LCT_GREY)
	{
		if (mode->bitdepth == 8)
		{
			*r = *g = *b = in[i];
			if (mode->key_defined && *r == mode->key_r) *a = 0;
			else *a = 255;
		}
		else if (mode->bitdepth == 16)
		{
			*r = *g = *b = in[i * 2 + 0];
			if (mode->key_defined && 256U * in[i * 2 + 0] + in[i * 2 + 1] == mode->key_r) *a = 0;
			else *a = 255;
		}
		else
		{
			unsigned highest = ((1U << mode->bitdepth) - 1U);
			size_t j = i * mode->bitdepth;
			unsigned value = readBitsFromReversedStream(&j, in, mode->bitdepth);
			*r = *g = *b = (value * 255) / highest;
			if (mode->key_defined && value == mode->key_r) *a = 0;
			else *a = 255;
		}
	}
	else if (mode->colortype == LCT_RGB)
	{
		if (mode->bitdepth == 8)
		{
			*r = in[i * 3 + 0]; *g = in[i * 3 + 1]; *b = in[i * 3 + 2];
			if (mode->key_defined && *r == mode->key_r && *g == mode->key_g && *b == mode->key_b) *a = 0;
			else *a = 255;
		}
		else
		{
			*r = in[i * 6 + 0];
			*g = in[i * 6 + 2];
			*b = in[i * 6 + 4];
			if (mode->key_defined && 256U * in[i * 6 + 0] + in[i * 6 + 1] == mode->key_r
				&& 256U * in[i * 6 + 2] + in[i * 6 + 3] == mode->key_g
				&& 256U * in[i * 6 + 4] + in[i * 6 + 5] == mode->key_b) *a = 0;
			else *a = 255;
		}
	}
	else if (mode->colortype == LCT_PALETTE)
	{
		unsigned index;
		if (mode->bitdepth == 8) index = in[i];
		else
		{
			size_t j = i * mode->bitdepth;
			index = readBitsFromReversedStream(&j, in, mode->bitdepth);
		}

		if (index >= mode->palettesize)
		{


			*r = *g = *b = 0;
			*a = 255;
		}
		else
		{
			*r = mode->palette[index * 4 + 0];
			*g = mode->palette[index * 4 + 1];
			*b = mode->palette[index * 4 + 2];
			*a = mode->palette[index * 4 + 3];
		}
	}
	else if (mode->colortype == LCT_GREY_ALPHA)
	{
		if (mode->bitdepth == 8)
		{
			*r = *g = *b = in[i * 2 + 0];
			*a = in[i * 2 + 1];
		}
		else
		{
			*r = *g = *b = in[i * 4 + 0];
			*a = in[i * 4 + 2];
		}
	}
	else if (mode->colortype == LCT_RGBA)
	{
		if (mode->bitdepth == 8)
		{
			*r = in[i * 4 + 0];
			*g = in[i * 4 + 1];
			*b = in[i * 4 + 2];
			*a = in[i * 4 + 3];
		}
		else
		{
			*r = in[i * 8 + 0];
			*g = in[i * 8 + 2];
			*b = in[i * 8 + 4];
			*a = in[i * 8 + 6];
		}
	}
}






static void getPixelColorsRGBA8(unsigned char* buffer, size_t numpixels,
	unsigned has_alpha, const unsigned char* in,
	const LodePNGColorMode* mode)
{
	unsigned num_channels = has_alpha ? 4 : 3;
	size_t i;
	if (mode->colortype == LCT_GREY)
	{
		if (mode->bitdepth == 8)
		{
			for (i = 0; i != numpixels; ++i, buffer += num_channels)
			{
				buffer[0] = buffer[1] = buffer[2] = in[i];
				if (has_alpha) buffer[3] = mode->key_defined && in[i] == mode->key_r ? 0 : 255;
			}
		}
		else if (mode->bitdepth == 16)
		{
			for (i = 0; i != numpixels; ++i, buffer += num_channels)
			{
				buffer[0] = buffer[1] = buffer[2] = in[i * 2];
				if (has_alpha) buffer[3] = mode->key_defined && 256U * in[i * 2 + 0] + in[i * 2 + 1] == mode->key_r ? 0 : 255;
			}
		}
		else
		{
			unsigned highest = ((1U << mode->bitdepth) - 1U);
			size_t j = 0;
			for (i = 0; i != numpixels; ++i, buffer += num_channels)
			{
				unsigned value = readBitsFromReversedStream(&j, in, mode->bitdepth);
				buffer[0] = buffer[1] = buffer[2] = (value * 255) / highest;
				if (has_alpha) buffer[3] = mode->key_defined && value == mode->key_r ? 0 : 255;
			}
		}
	}
	else if (mode->colortype == LCT_RGB)
	{
		if (mode->bitdepth == 8)
		{
			for (i = 0; i != numpixels; ++i, buffer += num_channels)
			{
				buffer[0] = in[i * 3 + 0];
				buffer[1] = in[i * 3 + 1];
				buffer[2] = in[i * 3 + 2];
				if (has_alpha) buffer[3] = mode->key_defined && buffer[0] == mode->key_r
					&& buffer[1] == mode->key_g && buffer[2] == mode->key_b ? 0 : 255;
			}
		}
		else
		{
			for (i = 0; i != numpixels; ++i, buffer += num_channels)
			{
				buffer[0] = in[i * 6 + 0];
				buffer[1] = in[i * 6 + 2];
				buffer[2] = in[i * 6 + 4];
				if (has_alpha) buffer[3] = mode->key_defined
					&& 256U * in[i * 6 + 0] + in[i * 6 + 1] == mode->key_r
					&& 256U * in[i * 6 + 2] + in[i * 6 + 3] == mode->key_g
					&& 256U * in[i * 6 + 4] + in[i * 6 + 5] == mode->key_b ? 0 : 255;
			}
		}
	}
	else if (mode->colortype == LCT_PALETTE)
	{
		unsigned index;
		size_t j = 0;
		for (i = 0; i != numpixels; ++i, buffer += num_channels)
		{
			if (mode->bitdepth == 8) index = in[i];
			else index = readBitsFromReversedStream(&j, in, mode->bitdepth);

			if (index >= mode->palettesize)
			{


				buffer[0] = buffer[1] = buffer[2] = 0;
				if (has_alpha) buffer[3] = 255;
			}
			else
			{
				buffer[0] = mode->palette[index * 4 + 0];
				buffer[1] = mode->palette[index * 4 + 1];
				buffer[2] = mode->palette[index * 4 + 2];
				if (has_alpha) buffer[3] = mode->palette[index * 4 + 3];
			}
		}
	}
	else if (mode->colortype == LCT_GREY_ALPHA)
	{
		if (mode->bitdepth == 8)
		{
			for (i = 0; i != numpixels; ++i, buffer += num_channels)
			{
				buffer[0] = buffer[1] = buffer[2] = in[i * 2 + 0];
				if (has_alpha) buffer[3] = in[i * 2 + 1];
			}
		}
		else
		{
			for (i = 0; i != numpixels; ++i, buffer += num_channels)
			{
				buffer[0] = buffer[1] = buffer[2] = in[i * 4 + 0];
				if (has_alpha) buffer[3] = in[i * 4 + 2];
			}
		}
	}
	else if (mode->colortype == LCT_RGBA)
	{
		if (mode->bitdepth == 8)
		{
			for (i = 0; i != numpixels; ++i, buffer += num_channels)
			{
				buffer[0] = in[i * 4 + 0];
				buffer[1] = in[i * 4 + 1];
				buffer[2] = in[i * 4 + 2];
				if (has_alpha) buffer[3] = in[i * 4 + 3];
			}
		}
		else
		{
			for (i = 0; i != numpixels; ++i, buffer += num_channels)
			{
				buffer[0] = in[i * 8 + 0];
				buffer[1] = in[i * 8 + 2];
				buffer[2] = in[i * 8 + 4];
				if (has_alpha) buffer[3] = in[i * 8 + 6];
			}
		}
	}
}



static void getPixelColorRGBA16(unsigned short* r, unsigned short* g, unsigned short* b, unsigned short* a,
	const unsigned char* in, size_t i, const LodePNGColorMode* mode)
{
	if (mode->colortype == LCT_GREY)
	{
		*r = *g = *b = 256 * in[i * 2 + 0] + in[i * 2 + 1];
		if (mode->key_defined && 256U * in[i * 2 + 0] + in[i * 2 + 1] == mode->key_r) *a = 0;
		else *a = 65535;
	}
	else if (mode->colortype == LCT_RGB)
	{
		*r = 256 * in[i * 6 + 0] + in[i * 6 + 1];
		*g = 256 * in[i * 6 + 2] + in[i * 6 + 3];
		*b = 256 * in[i * 6 + 4] + in[i * 6 + 5];
		if (mode->key_defined && 256U * in[i * 6 + 0] + in[i * 6 + 1] == mode->key_r
			&& 256U * in[i * 6 + 2] + in[i * 6 + 3] == mode->key_g
			&& 256U * in[i * 6 + 4] + in[i * 6 + 5] == mode->key_b) *a = 0;
		else *a = 65535;
	}
	else if (mode->colortype == LCT_GREY_ALPHA)
	{
		*r = *g = *b = 256 * in[i * 4 + 0] + in[i * 4 + 1];
		*a = 256 * in[i * 4 + 2] + in[i * 4 + 3];
	}
	else if (mode->colortype == LCT_RGBA)
	{
		*r = 256 * in[i * 8 + 0] + in[i * 8 + 1];
		*g = 256 * in[i * 8 + 2] + in[i * 8 + 3];
		*b = 256 * in[i * 8 + 4] + in[i * 8 + 5];
		*a = 256 * in[i * 8 + 6] + in[i * 8 + 7];
	}
}

unsigned lodepng_convert(unsigned char* out, const unsigned char* in,
	const LodePNGColorMode* mode_out, const LodePNGColorMode* mode_in,
	unsigned w, unsigned h)
{
	size_t i;
	ColorTree tree;
	size_t numpixels = w * h;

	if (lodepng_color_mode_equal(mode_out, mode_in))
	{
		size_t numbytes = lodepng_get_raw_size(w, h, mode_in);
		for (i = 0; i != numbytes; ++i) out[i] = in[i];
		return 0;
	}

	if (mode_out->colortype == LCT_PALETTE)
	{
		size_t palettesize = mode_out->palettesize;
		const unsigned char* palette = mode_out->palette;
		size_t palsize = 1u << mode_out->bitdepth;



		if (palettesize == 0) {
			palettesize = mode_in->palettesize;
			palette = mode_in->palette;
		}
		if (palettesize < palsize) palsize = palettesize;
		color_tree_init(&tree);
		for (i = 0; i != palsize; ++i)
		{
			const unsigned char* p = &palette[i * 4];
			color_tree_add(&tree, p[0], p[1], p[2], p[3], i);
		}
	}

	if (mode_in->bitdepth == 16 && mode_out->bitdepth == 16)
	{
		for (i = 0; i != numpixels; ++i)
		{
			unsigned short r = 0, g = 0, b = 0, a = 0;
			getPixelColorRGBA16(&r, &g, &b, &a, in, i, mode_in);
			rgba16ToPixel(out, i, mode_out, r, g, b, a);
		}
	}
	else if (mode_out->bitdepth == 8 && mode_out->colortype == LCT_RGBA)
	{
		getPixelColorsRGBA8(out, numpixels, 1, in, mode_in);
	}
	else if (mode_out->bitdepth == 8 && mode_out->colortype == LCT_RGB)
	{
		getPixelColorsRGBA8(out, numpixels, 0, in, mode_in);
	}
	else
	{
		unsigned char r = 0, g = 0, b = 0, a = 0;
		for (i = 0; i != numpixels; ++i)
		{
			getPixelColorRGBA8(&r, &g, &b, &a, in, i, mode_in);
			CERROR_TRY_RETURN(rgba8ToPixel(out, i, mode_out, &tree, r, g, b, a));
		}
	}

	if (mode_out->colortype == LCT_PALETTE)
	{
		color_tree_cleanup(&tree);
	}

	return 0;
}

#ifdef LODEPNG_COMPILE_ENCODER

void lodepng_color_profile_init(LodePNGColorProfile* profile)
{
	profile->colored = 0;
	profile->key = 0;
	profile->alpha = 0;
	profile->key_r = profile->key_g = profile->key_b = 0;
	profile->numcolors = 0;
	profile->bits = 1;
}
static unsigned getValueRequiredBits(unsigned char value)
{
	if (value == 0 || value == 255) return 1;

	if (value % 17 == 0) return value % 85 == 0 ? 2 : 4;
	return 8;
}



unsigned lodepng_get_color_profile(LodePNGColorProfile* profile,
	const unsigned char* in, unsigned w, unsigned h,
	const LodePNGColorMode* mode)
{
	unsigned error = 0;
	size_t i;
	ColorTree tree;
	size_t numpixels = w * h;

	unsigned colored_done = lodepng_is_greyscale_type(mode) ? 1 : 0;
	unsigned alpha_done = lodepng_can_have_alpha(mode) ? 0 : 1;
	unsigned numcolors_done = 0;
	unsigned bpp = lodepng_get_bpp(mode);
	unsigned bits_done = bpp == 1 ? 1 : 0;
	unsigned maxnumcolors = 257;
	unsigned sixteen = 0;
	if (bpp <= 8) maxnumcolors = bpp == 1 ? 2 : (bpp == 2 ? 4 : (bpp == 4 ? 16 : 256));

	color_tree_init(&tree);


	if (mode->bitdepth == 16)
	{
		unsigned short r, g, b, a;
		for (i = 0; i != numpixels; ++i)
		{
			getPixelColorRGBA16(&r, &g, &b, &a, in, i, mode);
			if ((r & 255) != ((r >> 8) & 255) || (g & 255) != ((g >> 8) & 255) ||
				(b & 255) != ((b >> 8) & 255) || (a & 255) != ((a >> 8) & 255))
			{
				sixteen = 1;
				break;
			}
		}
	}

	if (sixteen)
	{
		unsigned short r = 0, g = 0, b = 0, a = 0;
		profile->bits = 16;
		bits_done = numcolors_done = 1;

		for (i = 0; i != numpixels; ++i)
		{
			getPixelColorRGBA16(&r, &g, &b, &a, in, i, mode);

			if (!colored_done && (r != g || r != b))
			{
				profile->colored = 1;
				colored_done = 1;
			}

			if (!alpha_done)
			{
				unsigned matchkey = (r == profile->key_r && g == profile->key_g && b == profile->key_b);
				if (a != 65535 && (a != 0 || (profile->key && !matchkey)))
				{
					profile->alpha = 1;
					alpha_done = 1;
					if (profile->bits < 8) profile->bits = 8;
				}
				else if (a == 0 && !profile->alpha && !profile->key)
				{
					profile->key = 1;
					profile->key_r = r;
					profile->key_g = g;
					profile->key_b = b;
				}
				else if (a == 65535 && profile->key && matchkey)
				{

					profile->alpha = 1;
					alpha_done = 1;
				}
			}

			if (alpha_done && numcolors_done && colored_done && bits_done) break;
		}
	}
	else
	{
		for (i = 0; i != numpixels; ++i)
		{
			unsigned char r = 0, g = 0, b = 0, a = 0;
			getPixelColorRGBA8(&r, &g, &b, &a, in, i, mode);

			if (!bits_done && profile->bits < 8)
			{

				unsigned bits = getValueRequiredBits(r);
				if (bits > profile->bits) profile->bits = bits;
			}
			bits_done = (profile->bits >= bpp);

			if (!colored_done && (r != g || r != b))
			{
				profile->colored = 1;
				colored_done = 1;
				if (profile->bits < 8) profile->bits = 8;
			}

			if (!alpha_done)
			{
				unsigned matchkey = (r == profile->key_r && g == profile->key_g && b == profile->key_b);
				if (a != 255 && (a != 0 || (profile->key && !matchkey)))
				{
					profile->alpha = 1;
					alpha_done = 1;
					if (profile->bits < 8) profile->bits = 8;
				}
				else if (a == 0 && !profile->alpha && !profile->key)
				{
					profile->key = 1;
					profile->key_r = r;
					profile->key_g = g;
					profile->key_b = b;
				}
				else if (a == 255 && profile->key && matchkey)
				{

					profile->alpha = 1;
					alpha_done = 1;
					if (profile->bits < 8) profile->bits = 8;
				}
			}

			if (!numcolors_done)
			{
				if (!color_tree_has(&tree, r, g, b, a))
				{
					color_tree_add(&tree, r, g, b, a, profile->numcolors);
					if (profile->numcolors < 256)
					{
						unsigned char* p = profile->palette;
						unsigned n = profile->numcolors;
						p[n * 4 + 0] = r;
						p[n * 4 + 1] = g;
						p[n * 4 + 2] = b;
						p[n * 4 + 3] = a;
					}
					++profile->numcolors;
					numcolors_done = profile->numcolors >= maxnumcolors;
				}
			}

			if (alpha_done && numcolors_done && colored_done && bits_done) break;
		}


		profile->key_r += (profile->key_r << 8);
		profile->key_g += (profile->key_g << 8);
		profile->key_b += (profile->key_b << 8);
	}

	color_tree_cleanup(&tree);
	return error;
}






unsigned lodepng_auto_choose_color(LodePNGColorMode* mode_out,
	const unsigned char* image, unsigned w, unsigned h,
	const LodePNGColorMode* mode_in)
{
	LodePNGColorProfile prof;
	unsigned error = 0;
	unsigned i, n, palettebits, grey_ok, palette_ok;

	lodepng_color_profile_init(&prof);
	error = lodepng_get_color_profile(&prof, image, w, h, mode_in);
	if (error) return error;
	mode_out->key_defined = 0;

	if (prof.key && w * h <= 16)
	{
		prof.alpha = 1;
		if (prof.bits < 8) prof.bits = 8;
	}
	grey_ok = !prof.colored && !prof.alpha;
	n = prof.numcolors;
	palettebits = n <= 2 ? 1 : (n <= 4 ? 2 : (n <= 16 ? 4 : 8));
	palette_ok = n <= 256 && (n * 2 < w * h) && prof.bits <= 8;
	if (w * h < n * 2) palette_ok = 0;
	if (grey_ok && prof.bits <= palettebits) palette_ok = 0;

	if (palette_ok)
	{
		unsigned char* p = prof.palette;
		lodepng_palette_clear(mode_out);
		for (i = 0; i != prof.numcolors; ++i)
		{
			error = lodepng_palette_add(mode_out, p[i * 4 + 0], p[i * 4 + 1], p[i * 4 + 2], p[i * 4 + 3]);
			if (error) break;
		}

		mode_out->colortype = LCT_PALETTE;
		mode_out->bitdepth = palettebits;

		if (mode_in->colortype == LCT_PALETTE && mode_in->palettesize >= mode_out->palettesize
			&& mode_in->bitdepth == mode_out->bitdepth)
		{

			lodepng_color_mode_cleanup(mode_out);
			lodepng_color_mode_copy(mode_out, mode_in);
		}
	}
	else
	{
		mode_out->bitdepth = prof.bits;
		mode_out->colortype = prof.alpha ? (prof.colored ? LCT_RGBA : LCT_GREY_ALPHA)
			: (prof.colored ? LCT_RGB : LCT_GREY);

		if (prof.key && !prof.alpha)
		{
			unsigned mask = (1u << mode_out->bitdepth) - 1u;
			mode_out->key_r = prof.key_r & mask;
			mode_out->key_g = prof.key_g & mask;
			mode_out->key_b = prof.key_b & mask;
			mode_out->key_defined = 1;
		}
	}

	return error;
}

#endif






static unsigned char paethPredictor(short a, short b, short c)
{
	short pa = abs(b - c);
	short pb = abs(a - c);
	short pc = abs(a + b - c - c);

	if (pc < pa && pc < pb) return (unsigned char)c;
	else if (pb < pa) return (unsigned char)b;
	else return (unsigned char)a;
}



static const unsigned ADAM7_IX[7] = { 0, 4, 0, 2, 0, 1, 0 };
static const unsigned ADAM7_IY[7] = { 0, 0, 4, 0, 2, 0, 1 };
static const unsigned ADAM7_DX[7] = { 8, 8, 4, 4, 2, 2, 1 };
static const unsigned ADAM7_DY[7] = { 8, 8, 8, 4, 4, 2, 2 };
static void Adam7_getpassvalues(unsigned passw[7], unsigned passh[7], size_t filter_passstart[8],
	size_t padded_passstart[8], size_t passstart[8], unsigned w, unsigned h, unsigned bpp)
{

	unsigned i;


	for (i = 0; i != 7; ++i)
	{
		passw[i] = (w + ADAM7_DX[i] - ADAM7_IX[i] - 1) / ADAM7_DX[i];
		passh[i] = (h + ADAM7_DY[i] - ADAM7_IY[i] - 1) / ADAM7_DY[i];
		if (passw[i] == 0) passh[i] = 0;
		if (passh[i] == 0) passw[i] = 0;
	}

	filter_passstart[0] = padded_passstart[0] = passstart[0] = 0;
	for (i = 0; i != 7; ++i)
	{

		filter_passstart[i + 1] = filter_passstart[i]
			+ ((passw[i] && passh[i]) ? passh[i] * (1 + (passw[i] * bpp + 7) / 8) : 0);

		padded_passstart[i + 1] = padded_passstart[i] + passh[i] * ((passw[i] * bpp + 7) / 8);

		passstart[i + 1] = passstart[i] + (passh[i] * passw[i] * bpp + 7) / 8;
	}
}

#ifdef LODEPNG_COMPILE_DECODER






unsigned lodepng_inspect(unsigned* w, unsigned* h, LodePNGState* state,
	const unsigned char* in, size_t insize)
{
	LodePNGInfo* info = &state->info_png;
	if (insize == 0 || in == 0)
	{
		CERROR_RETURN_ERROR(state->error, 48);
	}
	if (insize < 33)
	{
		CERROR_RETURN_ERROR(state->error, 27);
	}


	lodepng_info_cleanup(info);
	lodepng_info_init(info);

	if (in[0] != 137 || in[1] != 80 || in[2] != 78 || in[3] != 71
		|| in[4] != 13 || in[5] != 10 || in[6] != 26 || in[7] != 10)
	{
		CERROR_RETURN_ERROR(state->error, 28);
	}
	if (in[12] != 'I' || in[13] != 'H' || in[14] != 'D' || in[15] != 'R')
	{
		CERROR_RETURN_ERROR(state->error, 29);
	}


	*w = lodepng_read32bitInt(&in[16]);
	*h = lodepng_read32bitInt(&in[20]);
	info->color.bitdepth = in[24];
	info->color.colortype = (LodePNGColorType)in[25];
	info->compression_method = in[26];
	info->filter_method = in[27];
	info->interlace_method = in[28];

	if (*w == 0 || *h == 0)
	{
		CERROR_RETURN_ERROR(state->error, 93);
	}

	if (!state->decoder.ignore_crc)
	{
		unsigned CRC = lodepng_read32bitInt(&in[29]);
		unsigned checksum = lodepng_crc32(&in[12], 17);
		if (CRC != checksum)
		{
			CERROR_RETURN_ERROR(state->error, 57);
		}
	}


	if (info->compression_method != 0) CERROR_RETURN_ERROR(state->error, 32);

	if (info->filter_method != 0) CERROR_RETURN_ERROR(state->error, 33);

	if (info->interlace_method > 1) CERROR_RETURN_ERROR(state->error, 34);

	state->error = checkColorValidity(info->color.colortype, info->color.bitdepth);
	return state->error;
}

static unsigned unfilterScanline(unsigned char* recon, const unsigned char* scanline, const unsigned char* precon,
	size_t bytewidth, unsigned char filterType, size_t length)
{
		size_t i;
	switch (filterType)
	{
	case 0:
		for (i = 0; i != length; ++i) recon[i] = scanline[i];
		break;
	case 1:
		for (i = 0; i != bytewidth; ++i) recon[i] = scanline[i];
		for (i = bytewidth; i < length; ++i) recon[i] = scanline[i] + recon[i - bytewidth];
		break;
	case 2:
		if (precon)
		{
			for (i = 0; i != length; ++i) recon[i] = scanline[i] + precon[i];
		}
		else
		{
			for (i = 0; i != length; ++i) recon[i] = scanline[i];
		}
		break;
	case 3:
		if (precon)
		{
			for (i = 0; i != bytewidth; ++i) recon[i] = scanline[i] + precon[i] / 2;
			for (i = bytewidth; i < length; ++i) recon[i] = scanline[i] + ((recon[i - bytewidth] + precon[i]) / 2);
		}
		else
		{
			for (i = 0; i != bytewidth; ++i) recon[i] = scanline[i];
			for (i = bytewidth; i < length; ++i) recon[i] = scanline[i] + recon[i - bytewidth] / 2;
		}
		break;
	case 4:
		if (precon)
		{
			for (i = 0; i != bytewidth; ++i)
			{
				recon[i] = (scanline[i] + precon[i]);
			}
			for (i = bytewidth; i < length; ++i)
			{
				recon[i] = (scanline[i] + paethPredictor(recon[i - bytewidth], precon[i], precon[i - bytewidth]));
			}
		}
		else
		{
			for (i = 0; i != bytewidth; ++i)
			{
				recon[i] = scanline[i];
			}
			for (i = bytewidth; i < length; ++i)
			{

				recon[i] = (scanline[i] + recon[i - bytewidth]);
			}
		}
		break;
	default: return 36;
	}
	return 0;
}

static unsigned unfilter(unsigned char* out, const unsigned char* in, unsigned w, unsigned h, unsigned bpp)
{
		unsigned y;
	unsigned char* prevline = 0;


	size_t bytewidth = (bpp + 7) / 8;
	size_t linebytes = (w * bpp + 7) / 8;

	for (y = 0; y < h; ++y)
	{
		size_t outindex = linebytes * y;
		size_t inindex = (1 + linebytes) * y;
		unsigned char filterType = in[inindex];

		CERROR_TRY_RETURN(unfilterScanline(&out[outindex], &in[inindex + 1], prevline, bytewidth, filterType, linebytes));

		prevline = &out[outindex];
	}

	return 0;
}
static void Adam7_deinterlace(unsigned char* out, const unsigned char* in, unsigned w, unsigned h, unsigned bpp)
{
	unsigned passw[7], passh[7];
	size_t filter_passstart[8], padded_passstart[8], passstart[8];
	unsigned i;

	Adam7_getpassvalues(passw, passh, filter_passstart, padded_passstart, passstart, w, h, bpp);

	if (bpp >= 8)
	{
		for (i = 0; i != 7; ++i)
		{
			unsigned x, y, b;
			size_t bytewidth = bpp / 8;
			for (y = 0; y < passh[i]; ++y)
				for (x = 0; x < passw[i]; ++x)
				{
					size_t pixelinstart = passstart[i] + (y * passw[i] + x) * bytewidth;
					size_t pixeloutstart = ((ADAM7_IY[i] + y * ADAM7_DY[i]) * w + ADAM7_IX[i] + x * ADAM7_DX[i]) * bytewidth;
					for (b = 0; b < bytewidth; ++b)
					{
						out[pixeloutstart + b] = in[pixelinstart + b];
					}
				}
		}
	}
	else
	{
		for (i = 0; i != 7; ++i)
		{
			unsigned x, y, b;
			unsigned ilinebits = bpp * passw[i];
			unsigned olinebits = bpp * w;
			size_t obp, ibp;
			for (y = 0; y < passh[i]; ++y)
				for (x = 0; x < passw[i]; ++x)
				{
					ibp = (8 * passstart[i]) + (y * ilinebits + x * bpp);
					obp = (ADAM7_IY[i] + y * ADAM7_DY[i]) * olinebits + (ADAM7_IX[i] + x * ADAM7_DX[i]) * bpp;
					for (b = 0; b < bpp; ++b)
					{
						unsigned char bit = readBitFromReversedStream(&ibp, in);

						setBitOfReversedStream0(&obp, out, bit);
					}
				}
		}
	}
}

static void removePaddingBits(unsigned char* out, const unsigned char* in,
	size_t olinebits, size_t ilinebits, unsigned h)
{
		unsigned y;
	size_t diff = ilinebits - olinebits;
	size_t ibp = 0, obp = 0;
	for (y = 0; y < h; ++y)
	{
		size_t x;
		for (x = 0; x < olinebits; ++x)
		{
			unsigned char bit = readBitFromReversedStream(&ibp, in);
			setBitOfReversedStream(&obp, out, bit);
		}
		ibp += diff;
	}
}




static unsigned postProcessScanlines(unsigned char* out, unsigned char* in,
	unsigned w, unsigned h, const LodePNGInfo* info_png)
{







	unsigned bpp = lodepng_get_bpp(&info_png->color);
	if (bpp == 0) return 31;

	if (info_png->interlace_method == 0)
	{
		if (bpp < 8 && w * bpp != ((w * bpp + 7) / 8) * 8)
		{
			CERROR_TRY_RETURN(unfilter(in, in, w, h, bpp));
			removePaddingBits(out, in, w * bpp, ((w * bpp + 7) / 8) * 8, h);
		}

		else CERROR_TRY_RETURN(unfilter(out, in, w, h, bpp));
	}
	else
	{
		unsigned passw[7], passh[7]; size_t filter_passstart[8], padded_passstart[8], passstart[8];
		unsigned i;

		Adam7_getpassvalues(passw, passh, filter_passstart, padded_passstart, passstart, w, h, bpp);

		for (i = 0; i != 7; ++i)
		{
			CERROR_TRY_RETURN(unfilter(&in[padded_passstart[i]], &in[filter_passstart[i]], passw[i], passh[i], bpp));


			if (bpp < 8)
			{


				removePaddingBits(&in[passstart[i]], &in[padded_passstart[i]], passw[i] * bpp,
					((passw[i] * bpp + 7) / 8) * 8, passh[i]);
			}
		}

		Adam7_deinterlace(out, in, w, h, bpp);
	}

	return 0;
}

static unsigned readChunk_PLTE(LodePNGColorMode* color, const unsigned char* data, size_t chunkLength)
{
	unsigned pos = 0, i;
	if (color->palette) lodepng_free(color->palette);
	color->palettesize = chunkLength / 3;
	color->palette = (unsigned char*)lodepng_malloc(4 * color->palettesize);
	if (!color->palette && color->palettesize)
	{
		color->palettesize = 0;
		return 83;
	}
	if (color->palettesize > 256) return 38;

	for (i = 0; i != color->palettesize; ++i)
	{
		color->palette[4 * i + 0] = data[pos++];
		color->palette[4 * i + 1] = data[pos++];
		color->palette[4 * i + 2] = data[pos++];
		color->palette[4 * i + 3] = 255;
	}

	return 0;
}

static unsigned readChunk_tRNS(LodePNGColorMode* color, const unsigned char* data, size_t chunkLength)
{
	unsigned i;
	if (color->colortype == LCT_PALETTE)
	{

		if (chunkLength > color->palettesize) return 38;

		for (i = 0; i != chunkLength; ++i) color->palette[4 * i + 3] = data[i];
	}
	else if (color->colortype == LCT_GREY)
	{

		if (chunkLength != 2) return 30;

		color->key_defined = 1;
		color->key_r = color->key_g = color->key_b = 256u * data[0] + data[1];
	}
	else if (color->colortype == LCT_RGB)
	{

		if (chunkLength != 6) return 41;

		color->key_defined = 1;
		color->key_r = 256u * data[0] + data[1];
		color->key_g = 256u * data[2] + data[3];
		color->key_b = 256u * data[4] + data[5];
	}
	else return 42;

	return 0;
}


#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS

static unsigned readChunk_bKGD(LodePNGInfo* info, const unsigned char* data, size_t chunkLength)
{
	if (info->color.colortype == LCT_PALETTE)
	{

		if (chunkLength != 1) return 43;

		info->background_defined = 1;
		info->background_r = info->background_g = info->background_b = data[0];
	}
	else if (info->color.colortype == LCT_GREY || info->color.colortype == LCT_GREY_ALPHA)
	{

		if (chunkLength != 2) return 44;

		info->background_defined = 1;
		info->background_r = info->background_g = info->background_b = 256u * data[0] + data[1];
	}
	else if (info->color.colortype == LCT_RGB || info->color.colortype == LCT_RGBA)
	{

		if (chunkLength != 6) return 45;

		info->background_defined = 1;
		info->background_r = 256u * data[0] + data[1];
		info->background_g = 256u * data[2] + data[3];
		info->background_b = 256u * data[4] + data[5];
	}

	return 0;
}


static unsigned readChunk_tEXt(LodePNGInfo* info, const unsigned char* data, size_t chunkLength)
{
	unsigned error = 0;
	char *key = 0, *str = 0;
	unsigned i;

	while (!error)
	{
		unsigned length, string2_begin;

		length = 0;
		while (length < chunkLength && data[length] != 0) ++length;


		if (length < 1 || length > 79) CERROR_BREAK(error, 89);

		key = (char*)lodepng_malloc(length + 1);
		if (!key) CERROR_BREAK(error, 83);

		key[length] = 0;
		for (i = 0; i != length; ++i) key[i] = (char)data[i];

		string2_begin = length + 1;

		length = chunkLength < string2_begin ? 0 : chunkLength - string2_begin;
		str = (char*)lodepng_malloc(length + 1);
		if (!str) CERROR_BREAK(error, 83);

		str[length] = 0;
		for (i = 0; i != length; ++i) str[i] = (char)data[string2_begin + i];

		error = lodepng_add_text(info, key, str);

		break;
	}

	lodepng_free(key);
	lodepng_free(str);

	return error;
}


static unsigned readChunk_zTXt(LodePNGInfo* info, const LodePNGDecompressSettings* zlibsettings,
	const unsigned char* data, size_t chunkLength)
{
	unsigned error = 0;
	unsigned i;

	unsigned length, string2_begin;
	char *key = 0;
	ucvector decoded;

	ucvector_init(&decoded);

	while (!error)
	{
		for (length = 0; length < chunkLength && data[length] != 0; ++length);
		if (length + 2 >= chunkLength) CERROR_BREAK(error, 75);
		if (length < 1 || length > 79) CERROR_BREAK(error, 89);

		key = (char*)lodepng_malloc(length + 1);
		if (!key) CERROR_BREAK(error, 83);

		key[length] = 0;
		for (i = 0; i != length; ++i) key[i] = (char)data[i];

		if (data[length + 1] != 0) CERROR_BREAK(error, 72);

		string2_begin = length + 2;
		if (string2_begin > chunkLength) CERROR_BREAK(error, 75);

		length = chunkLength - string2_begin;

		error = zlib_decompress(&decoded.data, &decoded.size,
			(unsigned char*)(&data[string2_begin]),
			length, zlibsettings);
		if (error) break;
		ucvector_push_back(&decoded, 0);

		error = lodepng_add_text(info, key, (char*)decoded.data);

		break;
	}

	lodepng_free(key);
	ucvector_cleanup(&decoded);

	return error;
}


static unsigned readChunk_iTXt(LodePNGInfo* info, const LodePNGDecompressSettings* zlibsettings,
	const unsigned char* data, size_t chunkLength)
{
	unsigned error = 0;
	unsigned i;

	unsigned length, begin, compressed;
	char *key = 0, *langtag = 0, *transkey = 0;
	ucvector decoded;
	ucvector_init(&decoded);

	while (!error)
	{


		if (chunkLength < 5) CERROR_BREAK(error, 30);


		for (length = 0; length < chunkLength && data[length] != 0; ++length);
		if (length + 3 >= chunkLength) CERROR_BREAK(error, 75);
		if (length < 1 || length > 79) CERROR_BREAK(error, 89);

		key = (char*)lodepng_malloc(length + 1);
		if (!key) CERROR_BREAK(error, 83);

		key[length] = 0;
		for (i = 0; i != length; ++i) key[i] = (char)data[i];


		compressed = data[length + 1];
		if (data[length + 2] != 0) CERROR_BREAK(error, 72);





		begin = length + 3;
		length = 0;
		for (i = begin; i < chunkLength && data[i] != 0; ++i) ++length;

		langtag = (char*)lodepng_malloc(length + 1);
		if (!langtag) CERROR_BREAK(error, 83);

		langtag[length] = 0;
		for (i = 0; i != length; ++i) langtag[i] = (char)data[begin + i];


		begin += length + 1;
		length = 0;
		for (i = begin; i < chunkLength && data[i] != 0; ++i) ++length;

		transkey = (char*)lodepng_malloc(length + 1);
		if (!transkey) CERROR_BREAK(error, 83);

		transkey[length] = 0;
		for (i = 0; i != length; ++i) transkey[i] = (char)data[begin + i];


		begin += length + 1;

		length = chunkLength < begin ? 0 : chunkLength - begin;

		if (compressed)
		{

			error = zlib_decompress(&decoded.data, &decoded.size,
				(unsigned char*)(&data[begin]),
				length, zlibsettings);
			if (error) break;
			if (decoded.allocsize < decoded.size) decoded.allocsize = decoded.size;
			ucvector_push_back(&decoded, 0);
		}
		else
		{
			if (!ucvector_resize(&decoded, length + 1)) CERROR_BREAK(error, 83);

			decoded.data[length] = 0;
			for (i = 0; i != length; ++i) decoded.data[i] = data[begin + i];
		}

		error = lodepng_add_itext(info, key, langtag, transkey, (char*)decoded.data);

		break;
	}

	lodepng_free(key);
	lodepng_free(langtag);
	lodepng_free(transkey);
	ucvector_cleanup(&decoded);

	return error;
}

static unsigned readChunk_tIME(LodePNGInfo* info, const unsigned char* data, size_t chunkLength)
{
	if (chunkLength != 7) return 73;

	info->time_defined = 1;
	info->time.year = 256u * data[0] + data[1];
	info->time.month = data[2];
	info->time.day = data[3];
	info->time.hour = data[4];
	info->time.minute = data[5];
	info->time.second = data[6];

	return 0;
}

static unsigned readChunk_pHYs(LodePNGInfo* info, const unsigned char* data, size_t chunkLength)
{
	if (chunkLength != 9) return 74;

	info->phys_defined = 1;
	info->phys_x = 16777216u * data[0] + 65536u * data[1] + 256u * data[2] + data[3];
	info->phys_y = 16777216u * data[4] + 65536u * data[5] + 256u * data[6] + data[7];
	info->phys_unit = data[8];

	return 0;
}
#endif


static void decodeGeneric(unsigned char** out, unsigned* w, unsigned* h,
	LodePNGState* state,
	const unsigned char* in, size_t insize)
{
	unsigned char IEND = 0;
	const unsigned char* chunk;
	size_t i;
	ucvector idat;
	ucvector scanlines;
	size_t predict;
	size_t numpixels;


	unsigned unknown = 0;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
	unsigned critical_pos = 1;
#endif


	*out = 0;

	state->error = lodepng_inspect(w, h, state, in, insize);
	if (state->error) return;

	numpixels = *w * *h;


	if (*h != 0 && numpixels / *h != *w) CERROR_RETURN(state->error, 92);


	if (numpixels > 268435455) CERROR_RETURN(state->error, 92);

	ucvector_init(&idat);
	chunk = &in[33];



	while (!IEND && !state->error)
	{
		unsigned chunkLength;
		const unsigned char* data;


		if ((size_t)((chunk - in) + 12) > insize || chunk < in) CERROR_BREAK(state->error, 30);


		chunkLength = lodepng_chunk_length(chunk);

		if (chunkLength > 2147483647) CERROR_BREAK(state->error, 63);

		if ((size_t)((chunk - in) + chunkLength + 12) > insize || (chunk + chunkLength + 12) < in)
		{
			CERROR_BREAK(state->error, 64);
		}

		data = lodepng_chunk_data_const(chunk);


		if (lodepng_chunk_type_equals(chunk, "IDAT"))
		{
			size_t oldsize = idat.size;
			if (!ucvector_resize(&idat, oldsize + chunkLength)) CERROR_BREAK(state->error, 83);
			for (i = 0; i != chunkLength; ++i) idat.data[oldsize + i] = data[i];
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
			critical_pos = 3;
#endif
		}

		else if (lodepng_chunk_type_equals(chunk, "IEND"))
		{
			IEND = 1;
		}

		else if (lodepng_chunk_type_equals(chunk, "PLTE"))
		{
			state->error = readChunk_PLTE(&state->info_png.color, data, chunkLength);
			if (state->error) break;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
			critical_pos = 2;
#endif
		}

		else if (lodepng_chunk_type_equals(chunk, "tRNS"))
		{
			state->error = readChunk_tRNS(&state->info_png.color, data, chunkLength);
			if (state->error) break;
		}
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS

		else if (lodepng_chunk_type_equals(chunk, "bKGD"))
		{
			state->error = readChunk_bKGD(&state->info_png, data, chunkLength);
			if (state->error) break;
		}

		else if (lodepng_chunk_type_equals(chunk, "tEXt"))
		{
			if (state->decoder.read_text_chunks)
			{
				state->error = readChunk_tEXt(&state->info_png, data, chunkLength);
				if (state->error) break;
			}
		}

		else if (lodepng_chunk_type_equals(chunk, "zTXt"))
		{
			if (state->decoder.read_text_chunks)
			{
				state->error = readChunk_zTXt(&state->info_png, &state->decoder.zlibsettings, data, chunkLength);
				if (state->error) break;
			}
		}

		else if (lodepng_chunk_type_equals(chunk, "iTXt"))
		{
			if (state->decoder.read_text_chunks)
			{
				state->error = readChunk_iTXt(&state->info_png, &state->decoder.zlibsettings, data, chunkLength);
				if (state->error) break;
			}
		}
		else if (lodepng_chunk_type_equals(chunk, "tIME"))
		{
			state->error = readChunk_tIME(&state->info_png, data, chunkLength);
			if (state->error) break;
		}
		else if (lodepng_chunk_type_equals(chunk, "pHYs"))
		{
			state->error = readChunk_pHYs(&state->info_png, data, chunkLength);
			if (state->error) break;
		}
#endif
		else
		{

			if (!lodepng_chunk_ancillary(chunk)) CERROR_BREAK(state->error, 69);

			unknown = 1;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
			if (state->decoder.remember_unknown_chunks)
			{
				state->error = lodepng_chunk_append(&state->info_png.unknown_chunks_data[critical_pos - 1],
					&state->info_png.unknown_chunks_size[critical_pos - 1], chunk);
				if (state->error) break;
			}
#endif
		}

		if (!state->decoder.ignore_crc && !unknown)
		{
			if (lodepng_chunk_check_crc(chunk)) CERROR_BREAK(state->error, 57);
		}

		if (!IEND) chunk = lodepng_chunk_next_const(chunk);
	}

	ucvector_init(&scanlines);


	if (state->info_png.interlace_method == 0)
	{

		predict = lodepng_get_raw_size_idat(*w, *h, &state->info_png.color) + *h;
	}
	else
	{

		const LodePNGColorMode* color = &state->info_png.color;
		predict = 0;
		predict += lodepng_get_raw_size_idat((*w + 7) / 8, (*h + 7) / 8, color) + (*h + 7) / 8;
		if (*w > 4) predict += lodepng_get_raw_size_idat((*w + 3) / 8, (*h + 7) / 8, color) + (*h + 7) / 8;
		predict += lodepng_get_raw_size_idat((*w + 3) / 4, (*h + 3) / 8, color) + (*h + 3) / 8;
		if (*w > 2) predict += lodepng_get_raw_size_idat((*w + 1) / 4, (*h + 3) / 4, color) + (*h + 3) / 4;
		predict += lodepng_get_raw_size_idat((*w + 1) / 2, (*h + 1) / 4, color) + (*h + 1) / 4;
		if (*w > 1) predict += lodepng_get_raw_size_idat((*w + 0) / 2, (*h + 1) / 2, color) + (*h + 1) / 2;
		predict += lodepng_get_raw_size_idat((*w + 0) / 1, (*h + 0) / 2, color) + (*h + 0) / 2;
	}
	if (!state->error && !ucvector_reserve(&scanlines, predict)) state->error = 83;
	if (!state->error)
	{
		state->error = zlib_decompress(&scanlines.data, &scanlines.size, idat.data,
			idat.size, &state->decoder.zlibsettings);
		if (!state->error && scanlines.size != predict) state->error = 91;
	}
	ucvector_cleanup(&idat);

	if (!state->error)
	{
		size_t outsize = lodepng_get_raw_size(*w, *h, &state->info_png.color);
		ucvector outv;
		ucvector_init(&outv);
		if (!ucvector_resizev(&outv, outsize, 0)) state->error = 83;
		if (!state->error) state->error = postProcessScanlines(outv.data, scanlines.data, *w, *h, &state->info_png);
		*out = outv.data;
	}
	ucvector_cleanup(&scanlines);
}

unsigned lodepng_decode(unsigned char** out, unsigned* w, unsigned* h,
	LodePNGState* state,
	const unsigned char* in, size_t insize)
{
	*out = 0;
	decodeGeneric(out, w, h, state, in, insize);
	if (state->error) return state->error;
	if (!state->decoder.color_convert || lodepng_color_mode_equal(&state->info_raw, &state->info_png.color))
	{



		if (!state->decoder.color_convert)
		{
			state->error = lodepng_color_mode_copy(&state->info_raw, &state->info_png.color);
			if (state->error) return state->error;
		}
	}
	else
	{

		unsigned char* data = *out;
		size_t outsize;



		if (!(state->info_raw.colortype == LCT_RGB || state->info_raw.colortype == LCT_RGBA)
			&& !(state->info_raw.bitdepth == 8))
		{
			return 56;
		}

		outsize = lodepng_get_raw_size(*w, *h, &state->info_raw);
		*out = (unsigned char*)lodepng_malloc(outsize);
		if (!(*out))
		{
			state->error = 83;
		}
		else state->error = lodepng_convert(*out, data, &state->info_raw,
			&state->info_png.color, *w, *h);
		lodepng_free(data);
	}
	return state->error;
}

unsigned lodepng_decode_memory(unsigned char** out, unsigned* w, unsigned* h, const unsigned char* in,
	size_t insize, LodePNGColorType colortype, unsigned bitdepth)
{
	unsigned error;
	LodePNGState state;
	lodepng_state_init(&state);
	state.info_raw.colortype = colortype;
	state.info_raw.bitdepth = bitdepth;
	error = lodepng_decode(out, w, h, &state, in, insize);
	lodepng_state_cleanup(&state);
	return error;
}

unsigned lodepng_decode32(unsigned char** out, unsigned* w, unsigned* h, const unsigned char* in, size_t insize)
{
	return lodepng_decode_memory(out, w, h, in, insize, LCT_RGBA, 8);
}

unsigned lodepng_decode24(unsigned char** out, unsigned* w, unsigned* h, const unsigned char* in, size_t insize)
{
	return lodepng_decode_memory(out, w, h, in, insize, LCT_RGB, 8);
}

#ifdef LODEPNG_COMPILE_DISK
unsigned lodepng_decode_file(unsigned char** out, unsigned* w, unsigned* h, const char* filename,
	LodePNGColorType colortype, unsigned bitdepth)
{
	unsigned char* buffer;
	size_t buffersize;
	unsigned error;
	error = lodepng_load_file(&buffer, &buffersize, filename);
	if (!error) error = lodepng_decode_memory(out, w, h, buffer, buffersize, colortype, bitdepth);
	lodepng_free(buffer);
	return error;
}

unsigned lodepng_decode32_file(unsigned char** out, unsigned* w, unsigned* h, const char* filename)
{
	return lodepng_decode_file(out, w, h, filename, LCT_RGBA, 8);
}

unsigned lodepng_decode24_file(unsigned char** out, unsigned* w, unsigned* h, const char* filename)
{
	return lodepng_decode_file(out, w, h, filename, LCT_RGB, 8);
}
#endif

void lodepng_decoder_settings_init(LodePNGDecoderSettings* settings)
{
	settings->color_convert = 1;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
	settings->read_text_chunks = 1;
	settings->remember_unknown_chunks = 0;
#endif
	settings->ignore_crc = 0;
	lodepng_decompress_settings_init(&settings->zlibsettings);
}

#endif

#if defined(LODEPNG_COMPILE_DECODER) || defined(LODEPNG_COMPILE_ENCODER)

void lodepng_state_init(LodePNGState* state)
{
#ifdef LODEPNG_COMPILE_DECODER
	lodepng_decoder_settings_init(&state->decoder);
#endif
#ifdef LODEPNG_COMPILE_ENCODER
	lodepng_encoder_settings_init(&state->encoder);
#endif
	lodepng_color_mode_init(&state->info_raw);
	lodepng_info_init(&state->info_png);
	state->error = 1;
}

void lodepng_state_cleanup(LodePNGState* state)
{
	lodepng_color_mode_cleanup(&state->info_raw);
	lodepng_info_cleanup(&state->info_png);
}

void lodepng_state_copy(LodePNGState* dest, const LodePNGState* source)
{
	lodepng_state_cleanup(dest);
	*dest = *source;
	lodepng_color_mode_init(&dest->info_raw);
	lodepng_info_init(&dest->info_png);
	dest->error = lodepng_color_mode_copy(&dest->info_raw, &source->info_raw); if (dest->error) return;
	dest->error = lodepng_info_copy(&dest->info_png, &source->info_png); if (dest->error) return;
}

#endif

#ifdef LODEPNG_COMPILE_ENCODER






static unsigned addChunk(ucvector* out, const char* chunkName, const unsigned char* data, size_t length)
{
	CERROR_TRY_RETURN(lodepng_chunk_create(&out->data, &out->size, (unsigned)length, chunkName, data));
	out->allocsize = out->size;
	return 0;
}

static void writeSignature(ucvector* out)
{

	ucvector_push_back(out, 137);
	ucvector_push_back(out, 80);
	ucvector_push_back(out, 78);
	ucvector_push_back(out, 71);
	ucvector_push_back(out, 13);
	ucvector_push_back(out, 10);
	ucvector_push_back(out, 26);
	ucvector_push_back(out, 10);
}

static unsigned addChunk_IHDR(ucvector* out, unsigned w, unsigned h,
	LodePNGColorType colortype, unsigned bitdepth, unsigned interlace_method)
{
	unsigned error = 0;
	ucvector header;
	ucvector_init(&header);

	lodepng_add32bitInt(&header, w);
	lodepng_add32bitInt(&header, h);
	ucvector_push_back(&header, (unsigned char)bitdepth);
	ucvector_push_back(&header, (unsigned char)colortype);
	ucvector_push_back(&header, 0);
	ucvector_push_back(&header, 0);
	ucvector_push_back(&header, interlace_method);

	error = addChunk(out, "IHDR", header.data, header.size);
	ucvector_cleanup(&header);

	return error;
}

static unsigned addChunk_PLTE(ucvector* out, const LodePNGColorMode* info)
{
	unsigned error = 0;
	size_t i;
	ucvector PLTE;
	ucvector_init(&PLTE);
	for (i = 0; i != info->palettesize * 4; ++i)
	{

		if (i % 4 != 3) ucvector_push_back(&PLTE, info->palette[i]);
	}
	error = addChunk(out, "PLTE", PLTE.data, PLTE.size);
	ucvector_cleanup(&PLTE);

	return error;
}

static unsigned addChunk_tRNS(ucvector* out, const LodePNGColorMode* info)
{
	unsigned error = 0;
	size_t i;
	ucvector tRNS;
	ucvector_init(&tRNS);
	if (info->colortype == LCT_PALETTE)
	{
		size_t amount = info->palettesize;

		for (i = info->palettesize; i != 0; --i)
		{
			if (info->palette[4 * (i - 1) + 3] == 255) --amount;
			else break;
		}

		for (i = 0; i != amount; ++i) ucvector_push_back(&tRNS, info->palette[4 * i + 3]);
	}
	else if (info->colortype == LCT_GREY)
	{
		if (info->key_defined)
		{
			ucvector_push_back(&tRNS, (unsigned char)(info->key_r / 256));
			ucvector_push_back(&tRNS, (unsigned char)(info->key_r % 256));
		}
	}
	else if (info->colortype == LCT_RGB)
	{
		if (info->key_defined)
		{
			ucvector_push_back(&tRNS, (unsigned char)(info->key_r / 256));
			ucvector_push_back(&tRNS, (unsigned char)(info->key_r % 256));
			ucvector_push_back(&tRNS, (unsigned char)(info->key_g / 256));
			ucvector_push_back(&tRNS, (unsigned char)(info->key_g % 256));
			ucvector_push_back(&tRNS, (unsigned char)(info->key_b / 256));
			ucvector_push_back(&tRNS, (unsigned char)(info->key_b % 256));
		}
	}

	error = addChunk(out, "tRNS", tRNS.data, tRNS.size);
	ucvector_cleanup(&tRNS);

	return error;
}

static unsigned addChunk_IDAT(ucvector* out, const unsigned char* data, size_t datasize,
	LodePNGCompressSettings* zlibsettings)
{
	ucvector zlibdata;
	unsigned error = 0;


	ucvector_init(&zlibdata);
	error = zlib_compress(&zlibdata.data, &zlibdata.size, data, datasize, zlibsettings);
	if (!error) error = addChunk(out, "IDAT", zlibdata.data, zlibdata.size);
	ucvector_cleanup(&zlibdata);

	return error;
}

static unsigned addChunk_IEND(ucvector* out)
{
	unsigned error = 0;
	error = addChunk(out, "IEND", 0, 0);
	return error;
}

#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS

static unsigned addChunk_tEXt(ucvector* out, const char* keyword, const char* textstring)
{
	unsigned error = 0;
	size_t i;
	ucvector text;
	ucvector_init(&text);
	for (i = 0; keyword[i] != 0; ++i) ucvector_push_back(&text, (unsigned char)keyword[i]);
	if (i < 1 || i > 79) return 89;
	ucvector_push_back(&text, 0);
	for (i = 0; textstring[i] != 0; ++i) ucvector_push_back(&text, (unsigned char)textstring[i]);
	error = addChunk(out, "tEXt", text.data, text.size);
	ucvector_cleanup(&text);

	return error;
}

static unsigned addChunk_zTXt(ucvector* out, const char* keyword, const char* textstring,
	LodePNGCompressSettings* zlibsettings)
{
	unsigned error = 0;
	ucvector data, compressed;
	size_t i, textsize = strlen(textstring);

	ucvector_init(&data);
	ucvector_init(&compressed);
	for (i = 0; keyword[i] != 0; ++i) ucvector_push_back(&data, (unsigned char)keyword[i]);
	if (i < 1 || i > 79) return 89;
	ucvector_push_back(&data, 0);
	ucvector_push_back(&data, 0);

	error = zlib_compress(&compressed.data, &compressed.size,
		(unsigned char*)textstring, textsize, zlibsettings);
	if (!error)
	{
		for (i = 0; i != compressed.size; ++i) ucvector_push_back(&data, compressed.data[i]);
		error = addChunk(out, "zTXt", data.data, data.size);
	}

	ucvector_cleanup(&compressed);
	ucvector_cleanup(&data);
	return error;
}

static unsigned addChunk_iTXt(ucvector* out, unsigned compressed, const char* keyword, const char* langtag,
	const char* transkey, const char* textstring, LodePNGCompressSettings* zlibsettings)
{
	unsigned error = 0;
	ucvector data;
	size_t i, textsize = strlen(textstring);

	ucvector_init(&data);

	for (i = 0; keyword[i] != 0; ++i) ucvector_push_back(&data, (unsigned char)keyword[i]);
	if (i < 1 || i > 79) return 89;
	ucvector_push_back(&data, 0);
	ucvector_push_back(&data, compressed ? 1 : 0);
	ucvector_push_back(&data, 0);
	for (i = 0; langtag[i] != 0; ++i) ucvector_push_back(&data, (unsigned char)langtag[i]);
	ucvector_push_back(&data, 0);
	for (i = 0; transkey[i] != 0; ++i) ucvector_push_back(&data, (unsigned char)transkey[i]);
	ucvector_push_back(&data, 0);

	if (compressed)
	{
		ucvector compressed_data;
		ucvector_init(&compressed_data);
		error = zlib_compress(&compressed_data.data, &compressed_data.size,
			(unsigned char*)textstring, textsize, zlibsettings);
		if (!error)
		{
			for (i = 0; i != compressed_data.size; ++i) ucvector_push_back(&data, compressed_data.data[i]);
		}
		ucvector_cleanup(&compressed_data);
	}
	else
	{
		for (i = 0; textstring[i] != 0; ++i) ucvector_push_back(&data, (unsigned char)textstring[i]);
	}

	if (!error) error = addChunk(out, "iTXt", data.data, data.size);
	ucvector_cleanup(&data);
	return error;
}

static unsigned addChunk_bKGD(ucvector* out, const LodePNGInfo* info)
{
	unsigned error = 0;
	ucvector bKGD;
	ucvector_init(&bKGD);
	if (info->color.colortype == LCT_GREY || info->color.colortype == LCT_GREY_ALPHA)
	{
		ucvector_push_back(&bKGD, (unsigned char)(info->background_r / 256));
		ucvector_push_back(&bKGD, (unsigned char)(info->background_r % 256));
	}
	else if (info->color.colortype == LCT_RGB || info->color.colortype == LCT_RGBA)
	{
		ucvector_push_back(&bKGD, (unsigned char)(info->background_r / 256));
		ucvector_push_back(&bKGD, (unsigned char)(info->background_r % 256));
		ucvector_push_back(&bKGD, (unsigned char)(info->background_g / 256));
		ucvector_push_back(&bKGD, (unsigned char)(info->background_g % 256));
		ucvector_push_back(&bKGD, (unsigned char)(info->background_b / 256));
		ucvector_push_back(&bKGD, (unsigned char)(info->background_b % 256));
	}
	else if (info->color.colortype == LCT_PALETTE)
	{
		ucvector_push_back(&bKGD, (unsigned char)(info->background_r % 256));
	}

	error = addChunk(out, "bKGD", bKGD.data, bKGD.size);
	ucvector_cleanup(&bKGD);

	return error;
}

static unsigned addChunk_tIME(ucvector* out, const LodePNGTime* time)
{
	unsigned error = 0;
	unsigned char* data = (unsigned char*)lodepng_malloc(7);
	if (!data) return 83;
	data[0] = (unsigned char)(time->year / 256);
	data[1] = (unsigned char)(time->year % 256);
	data[2] = (unsigned char)time->month;
	data[3] = (unsigned char)time->day;
	data[4] = (unsigned char)time->hour;
	data[5] = (unsigned char)time->minute;
	data[6] = (unsigned char)time->second;
	error = addChunk(out, "tIME", data, 7);
	lodepng_free(data);
	return error;
}

static unsigned addChunk_pHYs(ucvector* out, const LodePNGInfo* info)
{
	unsigned error = 0;
	ucvector data;
	ucvector_init(&data);

	lodepng_add32bitInt(&data, info->phys_x);
	lodepng_add32bitInt(&data, info->phys_y);
	ucvector_push_back(&data, info->phys_unit);

	error = addChunk(out, "pHYs", data.data, data.size);
	ucvector_cleanup(&data);

	return error;
}

#endif

static void filterScanline(unsigned char* out, const unsigned char* scanline, const unsigned char* prevline,
	size_t length, size_t bytewidth, unsigned char filterType)
{
	size_t i;
	switch (filterType)
	{
	case 0:
		for (i = 0; i != length; ++i) out[i] = scanline[i];
		break;
	case 1:
		for (i = 0; i != bytewidth; ++i) out[i] = scanline[i];
		for (i = bytewidth; i < length; ++i) out[i] = scanline[i] - scanline[i - bytewidth];
		break;
	case 2:
		if (prevline)
		{
			for (i = 0; i != length; ++i) out[i] = scanline[i] - prevline[i];
		}
		else
		{
			for (i = 0; i != length; ++i) out[i] = scanline[i];
		}
		break;
	case 3:
		if (prevline)
		{
			for (i = 0; i != bytewidth; ++i) out[i] = scanline[i] - prevline[i] / 2;
			for (i = bytewidth; i < length; ++i) out[i] = scanline[i] - ((scanline[i - bytewidth] + prevline[i]) / 2);
		}
		else
		{
			for (i = 0; i != bytewidth; ++i) out[i] = scanline[i];
			for (i = bytewidth; i < length; ++i) out[i] = scanline[i] - scanline[i - bytewidth] / 2;
		}
		break;
	case 4:
		if (prevline)
		{

			for (i = 0; i != bytewidth; ++i) out[i] = (scanline[i] - prevline[i]);
			for (i = bytewidth; i < length; ++i)
			{
				out[i] = (scanline[i] - paethPredictor(scanline[i - bytewidth], prevline[i], prevline[i - bytewidth]));
			}
		}
		else
		{
			for (i = 0; i != bytewidth; ++i) out[i] = scanline[i];

			for (i = bytewidth; i < length; ++i) out[i] = (scanline[i] - scanline[i - bytewidth]);
		}
		break;
	default: return;
	}
}


static float flog2(float f)
{
	float result = 0;
	while (f > 32) { result += 4; f /= 16; }
	while (f > 2) { ++result; f /= 2; }
	return result + 1.442695f * (f * f * f / 3 - 3 * f * f / 2 + 3 * f - 1.83333f);
}

static unsigned filter(unsigned char* out, const unsigned char* in, unsigned w, unsigned h,
	const LodePNGColorMode* info, const LodePNGEncoderSettings* settings)
{






	unsigned bpp = lodepng_get_bpp(info);

	size_t linebytes = (w * bpp + 7) / 8;

	size_t bytewidth = (bpp + 7) / 8;
	const unsigned char* prevline = 0;
	unsigned x, y;
	unsigned error = 0;
	LodePNGFilterStrategy strategy = settings->filter_strategy;
		if (settings->filter_palette_zero &&
			(info->colortype == LCT_PALETTE || info->bitdepth < 8)) strategy = LFS_ZERO;

	if (bpp == 0) return 31;

	if (strategy == LFS_ZERO)
	{
		for (y = 0; y != h; ++y)
		{
			size_t outindex = (1 + linebytes) * y;
			size_t inindex = linebytes * y;
			out[outindex] = 0;
			filterScanline(&out[outindex + 1], &in[inindex], prevline, linebytes, bytewidth, 0);
			prevline = &in[inindex];
		}
	}
	else if (strategy == LFS_MINSUM)
	{

		size_t sum[5];
		ucvector attempt[5];
		size_t smallest = 0;
		unsigned char type, bestType = 0;

		for (type = 0; type != 5; ++type)
		{
			ucvector_init(&attempt[type]);
			if (!ucvector_resize(&attempt[type], linebytes)) return 83;
		}

		if (!error)
		{
			for (y = 0; y != h; ++y)
			{

				for (type = 0; type != 5; ++type)
				{
					filterScanline(attempt[type].data, &in[y * linebytes], prevline, linebytes, bytewidth, type);


					sum[type] = 0;
					if (type == 0)
					{
						for (x = 0; x != linebytes; ++x) sum[type] += (unsigned char)(attempt[type].data[x]);
					}
					else
					{
						for (x = 0; x != linebytes; ++x)
						{



							unsigned char s = attempt[type].data[x];
							sum[type] += s < 128 ? s : (255U - s);
						}
					}


					if (type == 0 || sum[type] < smallest)
					{
						bestType = type;
						smallest = sum[type];
					}
				}

				prevline = &in[y * linebytes];


				out[y * (linebytes + 1)] = bestType;
				for (x = 0; x != linebytes; ++x) out[y * (linebytes + 1) + 1 + x] = attempt[bestType].data[x];
			}
		}

		for (type = 0; type != 5; ++type) ucvector_cleanup(&attempt[type]);
	}
	else if (strategy == LFS_ENTROPY)
	{
		float sum[5];
		ucvector attempt[5];
		float smallest = 0;
		unsigned type, bestType = 0;
		unsigned count[256];

		for (type = 0; type != 5; ++type)
		{
			ucvector_init(&attempt[type]);
			if (!ucvector_resize(&attempt[type], linebytes)) return 83;
		}

		for (y = 0; y != h; ++y)
		{

			for (type = 0; type != 5; ++type)
			{
				filterScanline(attempt[type].data, &in[y * linebytes], prevline, linebytes, bytewidth, type);
				for (x = 0; x != 256; ++x) count[x] = 0;
				for (x = 0; x != linebytes; ++x) ++count[attempt[type].data[x]];
				++count[type];
				sum[type] = 0;
				for (x = 0; x != 256; ++x)
				{
					float p = count[x] / (float)(linebytes + 1);
					sum[type] += count[x] == 0 ? 0 : flog2(1 / p) * p;
				}

				if (type == 0 || sum[type] < smallest)
				{
					bestType = type;
					smallest = sum[type];
				}
			}

			prevline = &in[y * linebytes];


			out[y * (linebytes + 1)] = bestType;
			for (x = 0; x != linebytes; ++x) out[y * (linebytes + 1) + 1 + x] = attempt[bestType].data[x];
		}

		for (type = 0; type != 5; ++type) ucvector_cleanup(&attempt[type]);
	}
	else if (strategy == LFS_PREDEFINED)
	{
		for (y = 0; y != h; ++y)
		{
			size_t outindex = (1 + linebytes) * y;
			size_t inindex = linebytes * y;
			unsigned char type = settings->predefined_filters[y];
			out[outindex] = type;
			filterScanline(&out[outindex + 1], &in[inindex], prevline, linebytes, bytewidth, type);
			prevline = &in[inindex];
		}
	}
	else if (strategy == LFS_BRUTE_FORCE)
	{



		size_t size[5];
		ucvector attempt[5];
		size_t smallest = 0;
		unsigned type = 0, bestType = 0;
		unsigned char* dummy;
		LodePNGCompressSettings zlibsettings = settings->zlibsettings;




		zlibsettings.btype = 1;


		zlibsettings.custom_zlib = 0;
		zlibsettings.custom_deflate = 0;
		for (type = 0; type != 5; ++type)
		{
			ucvector_init(&attempt[type]);
			ucvector_resize(&attempt[type], linebytes);
		}
		for (y = 0; y != h; ++y)
		{
			for (type = 0; type != 5; ++type)
			{
				unsigned testsize = attempt[type].size;


				filterScanline(attempt[type].data, &in[y * linebytes], prevline, linebytes, bytewidth, type);
				size[type] = 0;
				dummy = 0;
				zlib_compress(&dummy, &size[type], attempt[type].data, testsize, &zlibsettings);
				lodepng_free(dummy);

				if (type == 0 || size[type] < smallest)
				{
					bestType = type;
					smallest = size[type];
				}
			}
			prevline = &in[y * linebytes];
			out[y * (linebytes + 1)] = bestType;
			for (x = 0; x != linebytes; ++x) out[y * (linebytes + 1) + 1 + x] = attempt[bestType].data[x];
		}
		for (type = 0; type != 5; ++type) ucvector_cleanup(&attempt[type]);
	}
	else return 88;

	return error;
}

static void addPaddingBits(unsigned char* out, const unsigned char* in,
	size_t olinebits, size_t ilinebits, unsigned h)
{


	unsigned y;
	size_t diff = olinebits - ilinebits;
	size_t obp = 0, ibp = 0;
	for (y = 0; y != h; ++y)
	{
		size_t x;
		for (x = 0; x < ilinebits; ++x)
		{
			unsigned char bit = readBitFromReversedStream(&ibp, in);
			setBitOfReversedStream(&obp, out, bit);
		}


		for (x = 0; x != diff; ++x) setBitOfReversedStream(&obp, out, 0);
	}
}
static void Adam7_interlace(unsigned char* out, const unsigned char* in, unsigned w, unsigned h, unsigned bpp)
{
	unsigned passw[7], passh[7];
	size_t filter_passstart[8], padded_passstart[8], passstart[8];
	unsigned i;

	Adam7_getpassvalues(passw, passh, filter_passstart, padded_passstart, passstart, w, h, bpp);

	if (bpp >= 8)
	{
		for (i = 0; i != 7; ++i)
		{
			unsigned x, y, b;
			size_t bytewidth = bpp / 8;
			for (y = 0; y < passh[i]; ++y)
				for (x = 0; x < passw[i]; ++x)
				{
					size_t pixelinstart = ((ADAM7_IY[i] + y * ADAM7_DY[i]) * w + ADAM7_IX[i] + x * ADAM7_DX[i]) * bytewidth;
					size_t pixeloutstart = passstart[i] + (y * passw[i] + x) * bytewidth;
					for (b = 0; b < bytewidth; ++b)
					{
						out[pixeloutstart + b] = in[pixelinstart + b];
					}
				}
		}
	}
	else
	{
		for (i = 0; i != 7; ++i)
		{
			unsigned x, y, b;
			unsigned ilinebits = bpp * passw[i];
			unsigned olinebits = bpp * w;
			size_t obp, ibp;
			for (y = 0; y < passh[i]; ++y)
				for (x = 0; x < passw[i]; ++x)
				{
					ibp = (ADAM7_IY[i] + y * ADAM7_DY[i]) * olinebits + (ADAM7_IX[i] + x * ADAM7_DX[i]) * bpp;
					obp = (8 * passstart[i]) + (y * ilinebits + x * bpp);
					for (b = 0; b < bpp; ++b)
					{
						unsigned char bit = readBitFromReversedStream(&ibp, in);
						setBitOfReversedStream(&obp, out, bit);
					}
				}
		}
	}
}

static unsigned preProcessScanlines(unsigned char** out, size_t* outsize, const unsigned char* in,
	unsigned w, unsigned h,
	const LodePNGInfo* info_png, const LodePNGEncoderSettings* settings)
{
	unsigned bpp = lodepng_get_bpp(&info_png->color);
	unsigned error = 0;

	if (info_png->interlace_method == 0)
	{
		*outsize = h + (h * ((w * bpp + 7) / 8));
		*out = (unsigned char*)lodepng_malloc(*outsize);
		if (!(*out) && (*outsize)) error = 83;

		if (!error)
		{

			if (bpp < 8 && w * bpp != ((w * bpp + 7) / 8) * 8)
			{
				unsigned char* padded = (unsigned char*)lodepng_malloc(h * ((w * bpp + 7) / 8));
				if (!padded) error = 83;
				if (!error)
				{
					addPaddingBits(padded, in, ((w * bpp + 7) / 8) * 8, w * bpp, h);
					error = filter(*out, padded, w, h, &info_png->color, settings);
				}
				lodepng_free(padded);
			}
			else
			{

				error = filter(*out, in, w, h, &info_png->color, settings);
			}
		}
	}
	else
	{
		unsigned passw[7], passh[7];
		size_t filter_passstart[8], padded_passstart[8], passstart[8];
		unsigned char* adam7;

		Adam7_getpassvalues(passw, passh, filter_passstart, padded_passstart, passstart, w, h, bpp);

		*outsize = filter_passstart[7];
		*out = (unsigned char*)lodepng_malloc(*outsize);
		if (!(*out)) error = 83;

		adam7 = (unsigned char*)lodepng_malloc(passstart[7]);
		if (!adam7 && passstart[7]) error = 83;

		if (!error)
		{
			unsigned i;

			Adam7_interlace(adam7, in, w, h, bpp);
			for (i = 0; i != 7; ++i)
			{
				if (bpp < 8)
				{
					unsigned char* padded = (unsigned char*)lodepng_malloc(padded_passstart[i + 1] - padded_passstart[i]);
					if (!padded) ERROR_BREAK(83);
					addPaddingBits(padded, &adam7[passstart[i]],
						((passw[i] * bpp + 7) / 8) * 8, passw[i] * bpp, passh[i]);
					error = filter(&(*out)[filter_passstart[i]], padded,
						passw[i], passh[i], &info_png->color, settings);
					lodepng_free(padded);
				}
				else
				{
					error = filter(&(*out)[filter_passstart[i]], &adam7[padded_passstart[i]],
						passw[i], passh[i], &info_png->color, settings);
				}

				if (error) break;
			}
		}

		lodepng_free(adam7);
	}

	return error;
}

static unsigned getPaletteTranslucency(const unsigned char* palette, size_t palettesize)
{
	size_t i;
	unsigned key = 0;
	unsigned r = 0, g = 0, b = 0;
	for (i = 0; i != palettesize; ++i)
	{
		if (!key && palette[4 * i + 3] == 0)
		{
			r = palette[4 * i + 0]; g = palette[4 * i + 1]; b = palette[4 * i + 2];
			key = 1;
			i = (size_t)(-1);
		}
		else if (palette[4 * i + 3] != 255) return 2;

		else if (key && r == palette[i * 4 + 0] && g == palette[i * 4 + 1] && b == palette[i * 4 + 2]) return 2;
	}
	return key;
}

#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
static unsigned addUnknownChunks(ucvector* out, unsigned char* data, size_t datasize)
{
	unsigned char* inchunk = data;
	while ((size_t)(inchunk - data) < datasize)
	{
		CERROR_TRY_RETURN(lodepng_chunk_append(&out->data, &out->size, inchunk));
		out->allocsize = out->size;
		inchunk = lodepng_chunk_next(inchunk);
	}
	return 0;
}
#endif

unsigned lodepng_encode(unsigned char** out, size_t* outsize,
	const unsigned char* image, unsigned w, unsigned h,
	LodePNGState* state)
{
	LodePNGInfo info;
	ucvector outv;
	unsigned char* data = 0;
	size_t datasize = 0;


	*out = 0;
	*outsize = 0;
	state->error = 0;

	lodepng_info_init(&info);
	lodepng_info_copy(&info, &state->info_png);

	if ((info.color.colortype == LCT_PALETTE || state->encoder.force_palette)
		&& (info.color.palettesize == 0 || info.color.palettesize > 256))
	{
		state->error = 68;
		return state->error;
	}

	if (state->encoder.auto_convert)
	{
		state->error = lodepng_auto_choose_color(&info.color, image, w, h, &state->info_raw);
	}
	if (state->error) return state->error;

	if (state->encoder.zlibsettings.btype > 2)
	{
		CERROR_RETURN_ERROR(state->error, 61);
	}
	if (state->info_png.interlace_method > 1)
	{
		CERROR_RETURN_ERROR(state->error, 71);
	}

	state->error = checkColorValidity(info.color.colortype, info.color.bitdepth);
	if (state->error) return state->error;
	state->error = checkColorValidity(state->info_raw.colortype, state->info_raw.bitdepth);
	if (state->error) return state->error;

	if (!lodepng_color_mode_equal(&state->info_raw, &info.color))
	{
		unsigned char* converted;
		size_t size = (w * h * lodepng_get_bpp(&info.color) + 7) / 8;

		converted = (unsigned char*)lodepng_malloc(size);
		if (!converted && size) state->error = 83;
		if (!state->error)
		{
			state->error = lodepng_convert(converted, image, &info.color, &state->info_raw, w, h);
		}
		if (!state->error) preProcessScanlines(&data, &datasize, converted, w, h, &info, &state->encoder);
		lodepng_free(converted);
	}
	else preProcessScanlines(&data, &datasize, image, w, h, &info, &state->encoder);

	ucvector_init(&outv);
	while (!state->error)
	{
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
		size_t i;
#endif

		writeSignature(&outv);

		addChunk_IHDR(&outv, w, h, info.color.colortype, info.color.bitdepth, info.interlace_method);
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS

		if (info.unknown_chunks_data[0])
		{
			state->error = addUnknownChunks(&outv, info.unknown_chunks_data[0], info.unknown_chunks_size[0]);
			if (state->error) break;
		}
#endif

		if (info.color.colortype == LCT_PALETTE)
		{
			addChunk_PLTE(&outv, &info.color);
		}
		if (state->encoder.force_palette && (info.color.colortype == LCT_RGB || info.color.colortype == LCT_RGBA))
		{
			addChunk_PLTE(&outv, &info.color);
		}

		if (info.color.colortype == LCT_PALETTE && getPaletteTranslucency(info.color.palette, info.color.palettesize) != 0)
		{
			addChunk_tRNS(&outv, &info.color);
		}
		if ((info.color.colortype == LCT_GREY || info.color.colortype == LCT_RGB) && info.color.key_defined)
		{
			addChunk_tRNS(&outv, &info.color);
		}
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS

		if (info.background_defined) addChunk_bKGD(&outv, &info);

		if (info.phys_defined) addChunk_pHYs(&outv, &info);


		if (info.unknown_chunks_data[1])
		{
			state->error = addUnknownChunks(&outv, info.unknown_chunks_data[1], info.unknown_chunks_size[1]);
			if (state->error) break;
		}
#endif

		state->error = addChunk_IDAT(&outv, data, datasize, &state->encoder.zlibsettings);
		if (state->error) break;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS

		if (info.time_defined) addChunk_tIME(&outv, &info.time);

		for (i = 0; i != info.text_num; ++i)
		{
			if (strlen(info.text_keys[i]) > 79)
			{
				state->error = 66;
				break;
			}
			if (strlen(info.text_keys[i]) < 1)
			{
				state->error = 67;
				break;
			}
			if (state->encoder.text_compression)
			{
				addChunk_zTXt(&outv, info.text_keys[i], info.text_strings[i], &state->encoder.zlibsettings);
			}
			else
			{
				addChunk_tEXt(&outv, info.text_keys[i], info.text_strings[i]);
			}
		}

		if (state->encoder.add_id)
		{
			unsigned alread_added_id_text = 0;
			for (i = 0; i != info.text_num; ++i)
			{
				if (!strcmp(info.text_keys[i], "LodePNG"))
				{
					alread_added_id_text = 1;
					break;
				}
			}
			if (alread_added_id_text == 0)
			{
				addChunk_tEXt(&outv, "LodePNG", LODEPNG_VERSION_STRING);
			}
		}

		for (i = 0; i != info.itext_num; ++i)
		{
			if (strlen(info.itext_keys[i]) > 79)
			{
				state->error = 66;
				break;
			}
			if (strlen(info.itext_keys[i]) < 1)
			{
				state->error = 67;
				break;
			}
			addChunk_iTXt(&outv, state->encoder.text_compression,
				info.itext_keys[i], info.itext_langtags[i], info.itext_transkeys[i], info.itext_strings[i],
				&state->encoder.zlibsettings);
		}


		if (info.unknown_chunks_data[2])
		{
			state->error = addUnknownChunks(&outv, info.unknown_chunks_data[2], info.unknown_chunks_size[2]);
			if (state->error) break;
		}
#endif
		addChunk_IEND(&outv);

		break;
	}

	lodepng_info_cleanup(&info);
	lodepng_free(data);

	*out = outv.data;
	*outsize = outv.size;

	return state->error;
}

unsigned lodepng_encode_memory(unsigned char** out, size_t* outsize, const unsigned char* image,
	unsigned w, unsigned h, LodePNGColorType colortype, unsigned bitdepth)
{
	unsigned error;
	LodePNGState state;
	lodepng_state_init(&state);
	state.info_raw.colortype = colortype;
	state.info_raw.bitdepth = bitdepth;
	state.info_png.color.colortype = colortype;
	state.info_png.color.bitdepth = bitdepth;
	lodepng_encode(out, outsize, image, w, h, &state);
	error = state.error;
	lodepng_state_cleanup(&state);
	return error;
}

unsigned lodepng_encode32(unsigned char** out, size_t* outsize, const unsigned char* image, unsigned w, unsigned h)
{
	return lodepng_encode_memory(out, outsize, image, w, h, LCT_RGBA, 8);
}

unsigned lodepng_encode24(unsigned char** out, size_t* outsize, const unsigned char* image, unsigned w, unsigned h)
{
	return lodepng_encode_memory(out, outsize, image, w, h, LCT_RGB, 8);
}

#ifdef LODEPNG_COMPILE_DISK
unsigned lodepng_encode_file(const char* filename, const unsigned char* image, unsigned w, unsigned h,
	LodePNGColorType colortype, unsigned bitdepth)
{
	unsigned char* buffer;
	size_t buffersize;
	unsigned error = lodepng_encode_memory(&buffer, &buffersize, image, w, h, colortype, bitdepth);
	if (!error) error = lodepng_save_file(buffer, buffersize, filename);
	lodepng_free(buffer);
	return error;
}

unsigned lodepng_encode32_file(const char* filename, const unsigned char* image, unsigned w, unsigned h)
{
	return lodepng_encode_file(filename, image, w, h, LCT_RGBA, 8);
}

unsigned lodepng_encode24_file(const char* filename, const unsigned char* image, unsigned w, unsigned h)
{
	return lodepng_encode_file(filename, image, w, h, LCT_RGB, 8);
}
#endif

void lodepng_encoder_settings_init(LodePNGEncoderSettings* settings)
{
	lodepng_compress_settings_init(&settings->zlibsettings);
	settings->filter_palette_zero = 1;
	settings->filter_strategy = LFS_MINSUM;
	settings->auto_convert = 1;
	settings->force_palette = 0;
	settings->predefined_filters = 0;
#ifdef LODEPNG_COMPILE_ANCILLARY_CHUNKS
	settings->add_id = 0;
	settings->text_compression = 1;
#endif
}

#endif
#endif

#ifdef LODEPNG_COMPILE_ERROR_TEXT
const char* lodepng_error_text(unsigned code)
{
	switch (code)
	{
	case 0: return "no error, everything went ok";
	case 1: return "nothing done yet";
	case 10: return "end of input memory reached without huffman end code";
	case 11: return "error in code tree made it jump outside of huffman tree";
	case 13: return "problem while processing dynamic deflate block";
	case 14: return "problem while processing dynamic deflate block";
	case 15: return "problem while processing dynamic deflate block";
	case 16: return "unexisting code while processing dynamic deflate block";
	case 17: return "end of out buffer memory reached while inflating";
	case 18: return "invalid distance code while inflating";
	case 19: return "end of out buffer memory reached while inflating";
	case 20: return "invalid deflate block BTYPE encountered while decoding";
	case 21: return "NLEN is not ones complement of LEN in a deflate block";
	case 22: return "end of out buffer memory reached while inflating";
	case 23: return "end of in buffer memory reached while inflating";
	case 24: return "invalid FCHECK in zlib header";
	case 25: return "invalid compression method in zlib header";
	case 26: return "FDICT encountered in zlib header while it's not used for PNG";
	case 27: return "PNG file is smaller than a PNG header";
	case 28: return "incorrect PNG signature, it's no PNG or corrupted";
	case 29: return "first chunk is not the header chunk";
	case 30: return "chunk length too large, chunk broken off at end of file";
	case 31: return "illegal PNG color type or bpp";
	case 32: return "illegal PNG compression method";
	case 33: return "illegal PNG filter method";
	case 34: return "illegal PNG interlace method";
	case 35: return "chunk length of a chunk is too large or the chunk too small";
	case 36: return "illegal PNG filter type encountered";
	case 37: return "illegal bit depth for this color type given";
	case 38: return "the palette is too big";
	case 39: return "more palette alpha values given in tRNS chunk than there are colors in the palette";
	case 40: return "tRNS chunk has wrong size for greyscale image";
	case 41: return "tRNS chunk has wrong size for RGB image";
	case 42: return "tRNS chunk appeared while it was not allowed for this color type";
	case 43: return "bKGD chunk has wrong size for palette image";
	case 44: return "bKGD chunk has wrong size for greyscale image";
	case 45: return "bKGD chunk has wrong size for RGB image";
	case 48: return "empty input buffer given to decoder. Maybe caused by non-existing file?";
	case 49: return "jumped past memory while generating dynamic huffman tree";
	case 50: return "jumped past memory while generating dynamic huffman tree";
	case 51: return "jumped past memory while inflating huffman block";
	case 52: return "jumped past memory while inflating";
	case 53: return "size of zlib data too small";
	case 54: return "repeat symbol in tree while there was no value symbol yet";
	case 55: return "jumped past tree while generating huffman tree";
	case 56: return "given output image colortype or bitdepth not supported for color conversion";
	case 57: return "invalid CRC encountered (checking CRC can be disabled)";
	case 58: return "invalid ADLER32 encountered (checking ADLER32 can be disabled)";
	case 59: return "requested color conversion not supported";
	case 60: return "invalid window size given in the settings of the encoder (must be 0-32768)";
	case 61: return "invalid BTYPE given in the settings of the encoder (only 0, 1 and 2 are allowed)";
	case 62: return "conversion from color to greyscale not supported";
	case 63: return "length of a chunk too long, max allowed for PNG is 2147483647 bytes per chunk";
	case 64: return "the length of the END symbol 256 in the Huffman tree is 0";
	case 66: return "the length of a text chunk keyword given to the encoder is longer than the maximum of 79 bytes";
	case 67: return "the length of a text chunk keyword given to the encoder is smaller than the minimum of 1 byte";
	case 68: return "tried to encode a PLTE chunk with a palette that has less than 1 or more than 256 colors";
	case 69: return "unknown chunk type with 'critical' flag encountered by the decoder";
	case 71: return "unexisting interlace mode given to encoder (must be 0 or 1)";
	case 72: return "while decoding, unexisting compression method encountering in zTXt or iTXt chunk (it must be 0)";
	case 73: return "invalid tIME chunk size";
	case 74: return "invalid pHYs chunk size";
	case 75: return "no null termination char found while decoding text chunk";
	case 76: return "iTXt chunk too short to contain required bytes";
	case 77: return "integer overflow in buffer size";
	case 78: return "failed to open file for reading";
	case 79: return "failed to open file for writing";
	case 80: return "tried creating a tree of 0 symbols";
	case 81: return "lazy matching at pos 0 is impossible";
	case 82: return "color conversion to palette requested while a color isn't in palette";
	case 83: return "memory allocation failed";
	case 84: return "given image too small to contain all pixels to be encoded";
	case 86: return "impossible offset in lz77 encoding (internal bug)";
	case 87: return "must provide custom zlib function pointer if LODEPNG_COMPILE_ZLIB is not defined";
	case 88: return "invalid filter strategy given for LodePNGEncoderSettings.filter_strategy";
	case 89: return "text chunk keyword too short or long: must have size 1-79";
	case 90: return "windowsize must be a power of two";
	case 91: return "invalid decompressed idat size";
	case 92: return "too many pixels, not supported";
	case 93: return "zero width or height is invalid";
	}
	return "unknown error code";
}
#endif
#ifdef LODEPNG_COMPILE_CPP
namespace lodepng
{

#ifdef LODEPNG_COMPILE_DISK
	unsigned load_file(std::vector<unsigned char>& buffer, const std::string& filename)
	{
		std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
		if (!file) return 78;


		std::streamsize size = 0;
		if (file.seekg(0, std::ios::end).good()) size = file.tellg();
		if (file.seekg(0, std::ios::beg).good()) size -= file.tellg();


		buffer.resize(size_t(size));
		if (size > 0) file.read((char*)(&buffer[0]), size);

		return 0;
	}


	unsigned save_file(const std::vector<unsigned char>& buffer, const std::string& filename)
	{
		std::ofstream file(filename.c_str(), std::ios::out | std::ios::binary);
		if (!file) return 79;
		file.write(buffer.empty() ? 0 : (char*)&buffer[0], std::streamsize(buffer.size()));
		return 0;
	}
#endif

#ifdef LODEPNG_COMPILE_ZLIB
#ifdef LODEPNG_COMPILE_DECODER
	unsigned decompress(std::vector<unsigned char>& out, const unsigned char* in, size_t insize,
		const LodePNGDecompressSettings& settings)
	{
		unsigned char* buffer = 0;
		size_t buffersize = 0;
		unsigned error = zlib_decompress(&buffer, &buffersize, in, insize, &settings);
		if (buffer)
		{
			out.insert(out.end(), &buffer[0], &buffer[buffersize]);
			lodepng_free(buffer);
		}
		return error;
	}

	unsigned decompress(std::vector<unsigned char>& out, const std::vector<unsigned char>& in,
		const LodePNGDecompressSettings& settings)
	{
		return decompress(out, in.empty() ? 0 : &in[0], in.size(), settings);
	}
#endif

#ifdef LODEPNG_COMPILE_ENCODER
	unsigned compress(std::vector<unsigned char>& out, const unsigned char* in, size_t insize,
		const LodePNGCompressSettings& settings)
	{
		unsigned char* buffer = 0;
		size_t buffersize = 0;
		unsigned error = zlib_compress(&buffer, &buffersize, in, insize, &settings);
		if (buffer)
		{
			out.insert(out.end(), &buffer[0], &buffer[buffersize]);
			lodepng_free(buffer);
		}
		return error;
	}

	unsigned compress(std::vector<unsigned char>& out, const std::vector<unsigned char>& in,
		const LodePNGCompressSettings& settings)
	{
		return compress(out, in.empty() ? 0 : &in[0], in.size(), settings);
	}
#endif
#endif


#ifdef LODEPNG_COMPILE_PNG

	State::State()
	{
		lodepng_state_init(this);
	}

	State::State(const State& other)
	{
		lodepng_state_init(this);
		lodepng_state_copy(this, &other);
	}

	State::~State()
	{
		lodepng_state_cleanup(this);
	}

	State& State::operator=(const State& other)
	{
		lodepng_state_copy(this, &other);
		return *this;
	}

#ifdef LODEPNG_COMPILE_DECODER

	unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h, const unsigned char* in,
		size_t insize, LodePNGColorType colortype, unsigned bitdepth)
	{
		unsigned char* buffer;
		unsigned error = lodepng_decode_memory(&buffer, &w, &h, in, insize, colortype, bitdepth);
		if (buffer && !error)
		{
			State state;
			state.info_raw.colortype = colortype;
			state.info_raw.bitdepth = bitdepth;
			size_t buffersize = lodepng_get_raw_size(w, h, &state.info_raw);
			out.insert(out.end(), &buffer[0], &buffer[buffersize]);
			lodepng_free(buffer);
		}
		return error;
	}

	unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h,
		const std::vector<unsigned char>& in, LodePNGColorType colortype, unsigned bitdepth)
	{
		return decode(out, w, h, in.empty() ? 0 : &in[0], (unsigned)in.size(), colortype, bitdepth);
	}

	unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h,
		State& state,
		const unsigned char* in, size_t insize)
	{
		unsigned char* buffer = NULL;
		unsigned error = lodepng_decode(&buffer, &w, &h, &state, in, insize);
		if (buffer && !error)
		{
			size_t buffersize = lodepng_get_raw_size(w, h, &state.info_raw);
			out.insert(out.end(), &buffer[0], &buffer[buffersize]);
		}
		lodepng_free(buffer);
		return error;
	}

	unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h,
		State& state,
		const std::vector<unsigned char>& in)
	{
		return decode(out, w, h, state, in.empty() ? 0 : &in[0], in.size());
	}

#ifdef LODEPNG_COMPILE_DISK
	unsigned decode(std::vector<unsigned char>& out, unsigned& w, unsigned& h, const std::string& filename,
		LodePNGColorType colortype, unsigned bitdepth)
	{
		std::vector<unsigned char> buffer;
		unsigned error = load_file(buffer, filename);
		if (error) return error;
		return decode(out, w, h, buffer, colortype, bitdepth);
	}
#endif
#endif

#ifdef LODEPNG_COMPILE_ENCODER
	unsigned encode(std::vector<unsigned char>& out, const unsigned char* in, unsigned w, unsigned h,
		LodePNGColorType colortype, unsigned bitdepth)
	{
		unsigned char* buffer;
		size_t buffersize;
		unsigned error = lodepng_encode_memory(&buffer, &buffersize, in, w, h, colortype, bitdepth);
		if (buffer)
		{
			out.insert(out.end(), &buffer[0], &buffer[buffersize]);
			lodepng_free(buffer);
		}
		return error;
	}

	unsigned encode(std::vector<unsigned char>& out,
		const std::vector<unsigned char>& in, unsigned w, unsigned h,
		LodePNGColorType colortype, unsigned bitdepth)
	{
		if (lodepng_get_raw_size_lct(w, h, colortype, bitdepth) > in.size()) return 84;
		return encode(out, in.empty() ? 0 : &in[0], w, h, colortype, bitdepth);
	}

	unsigned encode(std::vector<unsigned char>& out,
		const unsigned char* in, unsigned w, unsigned h,
		State& state)
	{
		unsigned char* buffer;
		size_t buffersize;
		unsigned error = lodepng_encode(&buffer, &buffersize, in, w, h, &state);
		if (buffer)
		{
			out.insert(out.end(), &buffer[0], &buffer[buffersize]);
			lodepng_free(buffer);
		}
		return error;
	}

	unsigned encode(std::vector<unsigned char>& out,
		const std::vector<unsigned char>& in, unsigned w, unsigned h,
		State& state)
	{
		if (lodepng_get_raw_size(w, h, &state.info_raw) > in.size()) return 84;
		return encode(out, in.empty() ? 0 : &in[0], w, h, state);
	}

#ifdef LODEPNG_COMPILE_DISK
	unsigned encode(const std::string& filename,
		const unsigned char* in, unsigned w, unsigned h,
		LodePNGColorType colortype, unsigned bitdepth)
	{
		std::vector<unsigned char> buffer;
		unsigned error = encode(buffer, in, w, h, colortype, bitdepth);
		if (!error) error = save_file(buffer, filename);
		return error;
	}

	unsigned encode(const std::string& filename,
		const std::vector<unsigned char>& in, unsigned w, unsigned h,
		LodePNGColorType colortype, unsigned bitdepth)
	{
		if (lodepng_get_raw_size_lct(w, h, colortype, bitdepth) > in.size()) return 84;
		return encode(filename, in.empty() ? 0 : &in[0], w, h, colortype, bitdepth);
	}
#endif
#endif
#endif
}
#endif
class PNG
{
public:
	unsigned w, h;
	std::vector<unsigned char> data;
	PNG() {}
	PNG(unsigned w, unsigned h) { Create(w, h); }
	PNG(std::string file) { Load(file); }
	unsigned Load(std::string file)
	{
		Free();
		return lodepng::decode(data, w, h, file.c_str());
	}
	unsigned Save(std::string file)
	{
		return lodepng::encode(file.c_str(), data, w, h);
	}
	void Create(unsigned w, unsigned h)
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