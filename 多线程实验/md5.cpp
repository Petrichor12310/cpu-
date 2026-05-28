#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <cstdint>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define MD5_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define MD5_ALWAYS_INLINE inline
#endif

#define MD5_SIMD_OPT

using namespace std;
using namespace chrono;

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
Byte *StringProcess(string input, int *n_byte)
{
	// 将输入的字符串转换为Byte为单位的数组
	Byte *blocks = (Byte *)input.c_str();
	int length = input.length();

	// 计算原始消息长度（以比特为单位）
	int bitLength = length * 8;

	// paddingBits: 原始消息需要的padding长度（以bit为单位）
	// 对于给定的消息，将其补齐至length%512==448为止
	// 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
	int paddingBits = bitLength % 512;
	if (paddingBits > 448)
	{
		paddingBits = 512 - (paddingBits - 448);
	}
	else if (paddingBits < 448)
	{
		paddingBits = 448 - paddingBits;
	}
	else if (paddingBits == 448)
	{
		paddingBits = 512;
	}

	// 原始消息需要的padding长度（以Byte为单位）
	int paddingBytes = paddingBits / 8;
	// 创建最终的字节数组
	// length + paddingBytes + 8:
	// 1. length为原始消息的长度（bits）
	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
	int paddedLength = length + paddingBytes + 8;
	Byte *paddedMessage = new Byte[paddedLength];

	// 复制原始消息
	memcpy(paddedMessage, blocks, length);

	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
	// 所以第一个byte是0x80
	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

	// 添加消息长度（64比特，小端格式）
	for (int i = 0; i < 8; ++i)
	{
		// 特别注意此处应当将bitLength转换为uint64_t
		// 这里的length是原始消息的长度
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}

	// 验证长度是否满足要求。此时长度应当是512bit的倍数
	int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
	*n_byte = paddedLength;
	return paddedMessage;
}


/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
void MD5Hash(string input, bit32 *state)
{

	Byte *paddedMessage;
	int *messageLength = new int[1];
	for (int i = 0; i < 1; i += 1)
	{
		paddedMessage = StringProcess(input, &messageLength[i]);
		// cout<<messageLength[i]<<endl;
		assert(messageLength[i] == messageLength[0]);
	}
	int n_blocks = messageLength[0] / 64;

	// bit32* state= new bit32[4];
	state[0] = 0x67452301;
	state[1] = 0xefcdab89;
	state[2] = 0x98badcfe;
	state[3] = 0x10325476;

	// 逐block地更新state
	for (int i = 0; i < n_blocks; i += 1)
	{
		bit32 x[16];

		// 下面的处理，在理解上较为复杂
		for (int i1 = 0; i1 < 16; ++i1)
		{
			x[i1] = (paddedMessage[4 * i1 + i * 64]) |
					(paddedMessage[4 * i1 + 1 + i * 64] << 8) |
					(paddedMessage[4 * i1 + 2 + i * 64] << 16) |
					(paddedMessage[4 * i1 + 3 + i * 64] << 24);
		}

		bit32 a = state[0], b = state[1], c = state[2], d = state[3];

		auto start = system_clock::now();
		/* Round 1 */
		FF(a, b, c, d, x[0], s11, 0xd76aa478);
		FF(d, a, b, c, x[1], s12, 0xe8c7b756);
		FF(c, d, a, b, x[2], s13, 0x242070db);
		FF(b, c, d, a, x[3], s14, 0xc1bdceee);
		FF(a, b, c, d, x[4], s11, 0xf57c0faf);
		FF(d, a, b, c, x[5], s12, 0x4787c62a);
		FF(c, d, a, b, x[6], s13, 0xa8304613);
		FF(b, c, d, a, x[7], s14, 0xfd469501);
		FF(a, b, c, d, x[8], s11, 0x698098d8);
		FF(d, a, b, c, x[9], s12, 0x8b44f7af);
		FF(c, d, a, b, x[10], s13, 0xffff5bb1);
		FF(b, c, d, a, x[11], s14, 0x895cd7be);
		FF(a, b, c, d, x[12], s11, 0x6b901122);
		FF(d, a, b, c, x[13], s12, 0xfd987193);
		FF(c, d, a, b, x[14], s13, 0xa679438e);
		FF(b, c, d, a, x[15], s14, 0x49b40821);

		/* Round 2 */
		GG(a, b, c, d, x[1], s21, 0xf61e2562);
		GG(d, a, b, c, x[6], s22, 0xc040b340);
		GG(c, d, a, b, x[11], s23, 0x265e5a51);
		GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
		GG(a, b, c, d, x[5], s21, 0xd62f105d);
		GG(d, a, b, c, x[10], s22, 0x2441453);
		GG(c, d, a, b, x[15], s23, 0xd8a1e681);
		GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
		GG(a, b, c, d, x[9], s21, 0x21e1cde6);
		GG(d, a, b, c, x[14], s22, 0xc33707d6);
		GG(c, d, a, b, x[3], s23, 0xf4d50d87);
		GG(b, c, d, a, x[8], s24, 0x455a14ed);
		GG(a, b, c, d, x[13], s21, 0xa9e3e905);
		GG(d, a, b, c, x[2], s22, 0xfcefa3f8);
		GG(c, d, a, b, x[7], s23, 0x676f02d9);
		GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

		/* Round 3 */
		HH(a, b, c, d, x[5], s31, 0xfffa3942);
		HH(d, a, b, c, x[8], s32, 0x8771f681);
		HH(c, d, a, b, x[11], s33, 0x6d9d6122);
		HH(b, c, d, a, x[14], s34, 0xfde5380c);
		HH(a, b, c, d, x[1], s31, 0xa4beea44);
		HH(d, a, b, c, x[4], s32, 0x4bdecfa9);
		HH(c, d, a, b, x[7], s33, 0xf6bb4b60);
		HH(b, c, d, a, x[10], s34, 0xbebfbc70);
		HH(a, b, c, d, x[13], s31, 0x289b7ec6);
		HH(d, a, b, c, x[0], s32, 0xeaa127fa);
		HH(c, d, a, b, x[3], s33, 0xd4ef3085);
		HH(b, c, d, a, x[6], s34, 0x4881d05);
		HH(a, b, c, d, x[9], s31, 0xd9d4d039);
		HH(d, a, b, c, x[12], s32, 0xe6db99e5);
		HH(c, d, a, b, x[15], s33, 0x1fa27cf8);
		HH(b, c, d, a, x[2], s34, 0xc4ac5665);

		/* Round 4 */
		II(a, b, c, d, x[0], s41, 0xf4292244);
		II(d, a, b, c, x[7], s42, 0x432aff97);
		II(c, d, a, b, x[14], s43, 0xab9423a7);
		II(b, c, d, a, x[5], s44, 0xfc93a039);
		II(a, b, c, d, x[12], s41, 0x655b59c3);
		II(d, a, b, c, x[3], s42, 0x8f0ccc92);
		II(c, d, a, b, x[10], s43, 0xffeff47d);
		II(b, c, d, a, x[1], s44, 0x85845dd1);
		II(a, b, c, d, x[8], s41, 0x6fa87e4f);
		II(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		II(c, d, a, b, x[6], s43, 0xa3014314);
		II(b, c, d, a, x[13], s44, 0x4e0811a1);
		II(a, b, c, d, x[4], s41, 0xf7537e82);
		II(d, a, b, c, x[11], s42, 0xbd3af235);
		II(c, d, a, b, x[2], s43, 0x2ad7d2bb);
		II(b, c, d, a, x[9], s44, 0xeb86d391);

		state[0] += a;
		state[1] += b;
		state[2] += c;
		state[3] += d;
	}

	// 下面的处理，在理解上较为复杂
	for (int i = 0; i < 4; i++)
	{
		uint32_t value = state[i];
		state[i] = ((value & 0xff) << 24) |		 // 将最低字节移到最高位
				   ((value & 0xff00) << 8) |	 // 将次低字节左移
				   ((value & 0xff0000) >> 8) |	 // 将次高字节右移
				   ((value & 0xff000000) >> 24); // 将最高字节移到最低位
	}

	// 输出最终的hash结果
	// for (int i1 = 0; i1 < 4; i1 += 1)
	// {
	// 	cout << std::setw(8) << std::setfill('0') << hex << state[i1];
	// }
	// cout << endl;

	// 释放动态分配的内存
	// 实现SIMD并行算法的时候，也请记得及时回收内存！
	delete[] paddedMessage;
	delete[] messageLength;
}

#if defined(__ARM_NEON) || defined(__aarch64__)
namespace
{
	MD5_ALWAYS_INLINE uint32x4_t Set1U32N(uint32_t x)
	{
		return vdupq_n_u32(x);
	}

	MD5_ALWAYS_INLINE uint32x4_t Set4U32N(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3)
	{
		uint32x4_t v = vdupq_n_u32(0u);
		v = vsetq_lane_u32(x0, v, 0);
		v = vsetq_lane_u32(x1, v, 1);
		v = vsetq_lane_u32(x2, v, 2);
		v = vsetq_lane_u32(x3, v, 3);
		return v;
	}

	MD5_ALWAYS_INLINE uint32x4_t RotateLeft32N(uint32x4_t value, int shift)
	{
		const int32x4_t left = vdupq_n_s32(shift);
		const int32x4_t right = vdupq_n_s32(shift - 32);
		return vorrq_u32(vshlq_u32(value, left), vshlq_u32(value, right));
	}

	MD5_ALWAYS_INLINE uint32x4_t FvN(uint32x4_t x, uint32x4_t y, uint32x4_t z)
	{
		return vorrq_u32(vandq_u32(x, y), vbicq_u32(z, x));
	}

	MD5_ALWAYS_INLINE uint32x4_t GvN(uint32x4_t x, uint32x4_t y, uint32x4_t z)
	{
		return vorrq_u32(vandq_u32(x, z), vbicq_u32(y, z));
	}

	MD5_ALWAYS_INLINE uint32x4_t HvN(uint32x4_t x, uint32x4_t y, uint32x4_t z)
	{
		return veorq_u32(veorq_u32(x, y), z);
	}

	MD5_ALWAYS_INLINE uint32x4_t IvN(uint32x4_t x, uint32x4_t y, uint32x4_t z)
	{
		return veorq_u32(y, vorrq_u32(x, vmvnq_u32(z)));
	}

	MD5_ALWAYS_INLINE void FFvN(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d, uint32x4_t x, int s, uint32_t ac)
	{
		a = vaddq_u32(a, FvN(b, c, d));
		a = vaddq_u32(a, x);
		a = vaddq_u32(a, Set1U32N(ac));
		a = RotateLeft32N(a, s);
		a = vaddq_u32(a, b);
	}
    

	MD5_ALWAYS_INLINE void GGvN(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d, uint32x4_t x, int s, uint32_t ac)
	{
		a = vaddq_u32(a, GvN(b, c, d));
		a = vaddq_u32(a, x);
		a = vaddq_u32(a, Set1U32N(ac));
		a = RotateLeft32N(a, s);
		a = vaddq_u32(a, b);
	}

	MD5_ALWAYS_INLINE void HHvN(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d, uint32x4_t x, int s, uint32_t ac)
	{
		a = vaddq_u32(a, HvN(b, c, d));
		a = vaddq_u32(a, x);
		a = vaddq_u32(a, Set1U32N(ac));
		a = RotateLeft32N(a, s);
		a = vaddq_u32(a, b);
	}

	MD5_ALWAYS_INLINE void IIvN(uint32x4_t &a, uint32x4_t b, uint32x4_t c, uint32x4_t d, uint32x4_t x, int s, uint32_t ac)
	{
		a = vaddq_u32(a, IvN(b, c, d));
		a = vaddq_u32(a, x);
		a = vaddq_u32(a, Set1U32N(ac));
		a = RotateLeft32N(a, s);
		a = vaddq_u32(a, b);
	}

	MD5_ALWAYS_INLINE uint32_t LoadWordLittleEndianN(const Byte *block, int wordIndex)
	{
		const int base = wordIndex * 4;
		return static_cast<uint32_t>(block[base]) |
				   (static_cast<uint32_t>(block[base + 1]) << 8) |
				   (static_cast<uint32_t>(block[base + 2]) << 16) |
				   (static_cast<uint32_t>(block[base + 3]) << 24);
	}

	MD5_ALWAYS_INLINE uint32x4_t SelectByMaskN(uint32x4_t mask, uint32x4_t newValue, uint32x4_t oldValue)
	{
		return vbslq_u32(mask, newValue, oldValue);
	}

	MD5_ALWAYS_INLINE bit32 ByteSwap32N(bit32 value)
	{
		return ((value & 0xffu) << 24) |
			   ((value & 0xff00u) << 8) |
			   ((value & 0xff0000u) >> 8) |
			   ((value & 0xff000000u) >> 24);
	}

	MD5_ALWAYS_INLINE void BuildSingleBlockWordsN(const string *input, uint32_t words[16])
	{
		Byte block[64] = {0};
		size_t length = 0;
		if (input != nullptr)
		{
			length = input->size();
			if (length > 0)
			{
				memcpy(block, input->data(), length);
			}
		}

		block[length] = 0x80;
		const uint64_t bitLength = static_cast<uint64_t>(length) * 8u;
		for (int i = 0; i < 8; ++i)
		{
			block[56 + i] = static_cast<Byte>((bitLength >> (i * 8)) & 0xffu);
		}

		for (int i = 0; i < 16; ++i)
		{
			words[i] = LoadWordLittleEndianN(block, i);
		}
	}
}
#endif

MD5_SIMD_OPT void MD5HashSIMD4(const string *inputs, bit32 states[4][4], int count)
{
	if (count < 0)
	{
		count = 0;
	}
	if (count > 4)
	{
		count = 4;
	}

	for (int lane = count; lane < 4; ++lane)
	{
		for (int i = 0; i < 4; ++i)
		{
			states[lane][i] = 0;
		}
	}

#if defined(__ARM_NEON) || defined(__aarch64__)
	if (count == 0)
	{
		return;
	}

	bool allSingleBlock = true;
	for (int lane = 0; lane < count; ++lane)
	{
		if (inputs[lane].size() > 55)
		{
			allSingleBlock = false;
			break;
		}
	}

	if (allSingleBlock)
	{
		uint32_t laneWords[4][16];
		for (int lane = 0; lane < 4; ++lane)
		{
			const string *source = (lane < count) ? &inputs[lane] : nullptr;
			BuildSingleBlockWordsN(source, laneWords[lane]);
		}

		uint32x4_t x[16];
		for (int word = 0; word < 16; ++word)
		{
			x[word] = Set4U32N(laneWords[0][word], laneWords[1][word], laneWords[2][word], laneWords[3][word]);
		}

		uint32x4_t state0 = Set1U32N(0x67452301u);
		uint32x4_t state1 = Set1U32N(0xefcdab89u);
		uint32x4_t state2 = Set1U32N(0x98badcfeu);
		uint32x4_t state3 = Set1U32N(0x10325476u);

		uint32x4_t a = state0;
		uint32x4_t b = state1;
		uint32x4_t c = state2;
		uint32x4_t d = state3;

		/* Round 1 */
		FFvN(a, b, c, d, x[0], s11, 0xd76aa478);
		FFvN(d, a, b, c, x[1], s12, 0xe8c7b756);
		FFvN(c, d, a, b, x[2], s13, 0x242070db);
		FFvN(b, c, d, a, x[3], s14, 0xc1bdceee);
		FFvN(a, b, c, d, x[4], s11, 0xf57c0faf);
		FFvN(d, a, b, c, x[5], s12, 0x4787c62a);
		FFvN(c, d, a, b, x[6], s13, 0xa8304613);
		FFvN(b, c, d, a, x[7], s14, 0xfd469501);
		FFvN(a, b, c, d, x[8], s11, 0x698098d8);
		FFvN(d, a, b, c, x[9], s12, 0x8b44f7af);
		FFvN(c, d, a, b, x[10], s13, 0xffff5bb1);
		FFvN(b, c, d, a, x[11], s14, 0x895cd7be);
		FFvN(a, b, c, d, x[12], s11, 0x6b901122);
		FFvN(d, a, b, c, x[13], s12, 0xfd987193);
		FFvN(c, d, a, b, x[14], s13, 0xa679438e);
		FFvN(b, c, d, a, x[15], s14, 0x49b40821);

		/* Round 2 */
		GGvN(a, b, c, d, x[1], s21, 0xf61e2562);
		GGvN(d, a, b, c, x[6], s22, 0xc040b340);
		GGvN(c, d, a, b, x[11], s23, 0x265e5a51);
		GGvN(b, c, d, a, x[0], s24, 0xe9b6c7aa);
		GGvN(a, b, c, d, x[5], s21, 0xd62f105d);
		GGvN(d, a, b, c, x[10], s22, 0x02441453);
		GGvN(c, d, a, b, x[15], s23, 0xd8a1e681);
		GGvN(b, c, d, a, x[4], s24, 0xe7d3fbc8);
		GGvN(a, b, c, d, x[9], s21, 0x21e1cde6);
		GGvN(d, a, b, c, x[14], s22, 0xc33707d6);
		GGvN(c, d, a, b, x[3], s23, 0xf4d50d87);
		GGvN(b, c, d, a, x[8], s24, 0x455a14ed);
		GGvN(a, b, c, d, x[13], s21, 0xa9e3e905);
		GGvN(d, a, b, c, x[2], s22, 0xfcefa3f8);
		GGvN(c, d, a, b, x[7], s23, 0x676f02d9);
		GGvN(b, c, d, a, x[12], s24, 0x8d2a4c8a);

		/* Round 3 */
		HHvN(a, b, c, d, x[5], s31, 0xfffa3942);
		HHvN(d, a, b, c, x[8], s32, 0x8771f681);
		HHvN(c, d, a, b, x[11], s33, 0x6d9d6122);
		HHvN(b, c, d, a, x[14], s34, 0xfde5380c);
		HHvN(a, b, c, d, x[1], s31, 0xa4beea44);
		HHvN(d, a, b, c, x[4], s32, 0x4bdecfa9);
		HHvN(c, d, a, b, x[7], s33, 0xf6bb4b60);
		HHvN(b, c, d, a, x[10], s34, 0xbebfbc70);
		HHvN(a, b, c, d, x[13], s31, 0x289b7ec6);
		HHvN(d, a, b, c, x[0], s32, 0xeaa127fa);
		HHvN(c, d, a, b, x[3], s33, 0xd4ef3085);
		HHvN(b, c, d, a, x[6], s34, 0x04881d05);
		HHvN(a, b, c, d, x[9], s31, 0xd9d4d039);
		HHvN(d, a, b, c, x[12], s32, 0xe6db99e5);
		HHvN(c, d, a, b, x[15], s33, 0x1fa27cf8);
		HHvN(b, c, d, a, x[2], s34, 0xc4ac5665);

		/* Round 4 */
		IIvN(a, b, c, d, x[0], s41, 0xf4292244);
		IIvN(d, a, b, c, x[7], s42, 0x432aff97);
		IIvN(c, d, a, b, x[14], s43, 0xab9423a7);
		IIvN(b, c, d, a, x[5], s44, 0xfc93a039);
		IIvN(a, b, c, d, x[12], s41, 0x655b59c3);
		IIvN(d, a, b, c, x[3], s42, 0x8f0ccc92);
		IIvN(c, d, a, b, x[10], s43, 0xffeff47d);
		IIvN(b, c, d, a, x[1], s44, 0x85845dd1);
		IIvN(a, b, c, d, x[8], s41, 0x6fa87e4f);
		IIvN(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		IIvN(c, d, a, b, x[6], s43, 0xa3014314);
		IIvN(b, c, d, a, x[13], s44, 0x4e0811a1);
		IIvN(a, b, c, d, x[4], s41, 0xf7537e82);
		IIvN(d, a, b, c, x[11], s42, 0xbd3af235);
		IIvN(c, d, a, b, x[2], s43, 0x2ad7d2bb);
		IIvN(b, c, d, a, x[9], s44, 0xeb86d391);

		state0 = vaddq_u32(state0, a);
		state1 = vaddq_u32(state1, b);
		state2 = vaddq_u32(state2, c);
		state3 = vaddq_u32(state3, d);

		uint32_t out0[4];
		uint32_t out1[4];
		uint32_t out2[4];
		uint32_t out3[4];
		vst1q_u32(out0, state0);
		vst1q_u32(out1, state1);
		vst1q_u32(out2, state2);
		vst1q_u32(out3, state3);

		for (int lane = 0; lane < count; ++lane)
		{
			states[lane][0] = ByteSwap32N(out0[lane]);
			states[lane][1] = ByteSwap32N(out1[lane]);
			states[lane][2] = ByteSwap32N(out2[lane]);
			states[lane][3] = ByteSwap32N(out3[lane]);
		}
		return;
	}

	Byte *paddedMessages[4] = {nullptr, nullptr, nullptr, nullptr};
	int messageBytes[4] = {0, 0, 0, 0};
	int nBlocks[4] = {0, 0, 0, 0};
	int maxBlocks = 0;

	for (int lane = 0; lane < count; ++lane)
	{
		paddedMessages[lane] = StringProcess(inputs[lane], &messageBytes[lane]);
		nBlocks[lane] = messageBytes[lane] / 64;
		if (nBlocks[lane] > maxBlocks)
		{
			maxBlocks = nBlocks[lane];
		}
	}

	uint32x4_t state0 = Set1U32N(0x67452301u);
	uint32x4_t state1 = Set1U32N(0xefcdab89u);
	uint32x4_t state2 = Set1U32N(0x98badcfeu);
	uint32x4_t state3 = Set1U32N(0x10325476u);

	for (int block = 0; block < maxBlocks; ++block)
	{
		uint32_t activeLaneMask[4] = {0u, 0u, 0u, 0u};
		for (int lane = 0; lane < count; ++lane)
		{
			if (block < nBlocks[lane])
			{
				activeLaneMask[lane] = 0xffffffffu;
			}
		}

		uint32x4_t activeMask = Set4U32N(activeLaneMask[0], activeLaneMask[1], activeLaneMask[2], activeLaneMask[3]);

		uint32x4_t x[16];
		for (int word = 0; word < 16; ++word)
		{
			uint32_t laneWords[4] = {0u, 0u, 0u, 0u};
			for (int lane = 0; lane < count; ++lane)
			{
				if (activeLaneMask[lane] != 0u)
				{
					const Byte *blockPtr = paddedMessages[lane] + block * 64;
					laneWords[lane] = LoadWordLittleEndianN(blockPtr, word);
				}
			}
			x[word] = Set4U32N(laneWords[0], laneWords[1], laneWords[2], laneWords[3]);
		}

		uint32x4_t a = state0;
		uint32x4_t b = state1;
		uint32x4_t c = state2;
		uint32x4_t d = state3;

		/* Round 1 */
		FFvN(a, b, c, d, x[0], s11, 0xd76aa478);
		FFvN(d, a, b, c, x[1], s12, 0xe8c7b756);
		FFvN(c, d, a, b, x[2], s13, 0x242070db);
		FFvN(b, c, d, a, x[3], s14, 0xc1bdceee);
		FFvN(a, b, c, d, x[4], s11, 0xf57c0faf);
		FFvN(d, a, b, c, x[5], s12, 0x4787c62a);
		FFvN(c, d, a, b, x[6], s13, 0xa8304613);
		FFvN(b, c, d, a, x[7], s14, 0xfd469501);
		FFvN(a, b, c, d, x[8], s11, 0x698098d8);
		FFvN(d, a, b, c, x[9], s12, 0x8b44f7af);
		FFvN(c, d, a, b, x[10], s13, 0xffff5bb1);
		FFvN(b, c, d, a, x[11], s14, 0x895cd7be);
		FFvN(a, b, c, d, x[12], s11, 0x6b901122);
		FFvN(d, a, b, c, x[13], s12, 0xfd987193);
		FFvN(c, d, a, b, x[14], s13, 0xa679438e);
		FFvN(b, c, d, a, x[15], s14, 0x49b40821);

		/* Round 2 */
		GGvN(a, b, c, d, x[1], s21, 0xf61e2562);
		GGvN(d, a, b, c, x[6], s22, 0xc040b340);
		GGvN(c, d, a, b, x[11], s23, 0x265e5a51);
		GGvN(b, c, d, a, x[0], s24, 0xe9b6c7aa);
		GGvN(a, b, c, d, x[5], s21, 0xd62f105d);
		GGvN(d, a, b, c, x[10], s22, 0x02441453);
		GGvN(c, d, a, b, x[15], s23, 0xd8a1e681);
		GGvN(b, c, d, a, x[4], s24, 0xe7d3fbc8);
		GGvN(a, b, c, d, x[9], s21, 0x21e1cde6);
		GGvN(d, a, b, c, x[14], s22, 0xc33707d6);
		GGvN(c, d, a, b, x[3], s23, 0xf4d50d87);
		GGvN(b, c, d, a, x[8], s24, 0x455a14ed);
		GGvN(a, b, c, d, x[13], s21, 0xa9e3e905);
		GGvN(d, a, b, c, x[2], s22, 0xfcefa3f8);
		GGvN(c, d, a, b, x[7], s23, 0x676f02d9);
		GGvN(b, c, d, a, x[12], s24, 0x8d2a4c8a);

		/* Round 3 */
		HHvN(a, b, c, d, x[5], s31, 0xfffa3942);
		HHvN(d, a, b, c, x[8], s32, 0x8771f681);
		HHvN(c, d, a, b, x[11], s33, 0x6d9d6122);
		HHvN(b, c, d, a, x[14], s34, 0xfde5380c);
		HHvN(a, b, c, d, x[1], s31, 0xa4beea44);
		HHvN(d, a, b, c, x[4], s32, 0x4bdecfa9);
		HHvN(c, d, a, b, x[7], s33, 0xf6bb4b60);
		HHvN(b, c, d, a, x[10], s34, 0xbebfbc70);
		HHvN(a, b, c, d, x[13], s31, 0x289b7ec6);
		HHvN(d, a, b, c, x[0], s32, 0xeaa127fa);
		HHvN(c, d, a, b, x[3], s33, 0xd4ef3085);
		HHvN(b, c, d, a, x[6], s34, 0x04881d05);
		HHvN(a, b, c, d, x[9], s31, 0xd9d4d039);
		HHvN(d, a, b, c, x[12], s32, 0xe6db99e5);
		HHvN(c, d, a, b, x[15], s33, 0x1fa27cf8);
		HHvN(b, c, d, a, x[2], s34, 0xc4ac5665);

		/* Round 4 */
		IIvN(a, b, c, d, x[0], s41, 0xf4292244);
		IIvN(d, a, b, c, x[7], s42, 0x432aff97);
		IIvN(c, d, a, b, x[14], s43, 0xab9423a7);
		IIvN(b, c, d, a, x[5], s44, 0xfc93a039);
		IIvN(a, b, c, d, x[12], s41, 0x655b59c3);
		IIvN(d, a, b, c, x[3], s42, 0x8f0ccc92);
		IIvN(c, d, a, b, x[10], s43, 0xffeff47d);
		IIvN(b, c, d, a, x[1], s44, 0x85845dd1);
		IIvN(a, b, c, d, x[8], s41, 0x6fa87e4f);
		IIvN(d, a, b, c, x[15], s42, 0xfe2ce6e0);
		IIvN(c, d, a, b, x[6], s43, 0xa3014314);
		IIvN(b, c, d, a, x[13], s44, 0x4e0811a1);
		IIvN(a, b, c, d, x[4], s41, 0xf7537e82);
		IIvN(d, a, b, c, x[11], s42, 0xbd3af235);
		IIvN(c, d, a, b, x[2], s43, 0x2ad7d2bb);
		IIvN(b, c, d, a, x[9], s44, 0xeb86d391);

		uint32x4_t next0 = vaddq_u32(state0, a);
		uint32x4_t next1 = vaddq_u32(state1, b);
		uint32x4_t next2 = vaddq_u32(state2, c);
		uint32x4_t next3 = vaddq_u32(state3, d);

		state0 = SelectByMaskN(activeMask, next0, state0);
		state1 = SelectByMaskN(activeMask, next1, state1);
		state2 = SelectByMaskN(activeMask, next2, state2);
		state3 = SelectByMaskN(activeMask, next3, state3);
	}

	uint32_t out0[4];
	uint32_t out1[4];
	uint32_t out2[4];
	uint32_t out3[4];
	vst1q_u32(out0, state0);
	vst1q_u32(out1, state1);
	vst1q_u32(out2, state2);
	vst1q_u32(out3, state3);

	for (int lane = 0; lane < count; ++lane)
	{
		states[lane][0] = ByteSwap32N(out0[lane]);
		states[lane][1] = ByteSwap32N(out1[lane]);
		states[lane][2] = ByteSwap32N(out2[lane]);
		states[lane][3] = ByteSwap32N(out3[lane]);
	}

	for (int lane = 0; lane < count; ++lane)
	{
		delete[] paddedMessages[lane];
	}
#else
	for (int lane = 0; lane < count; ++lane)
	{
		MD5Hash(inputs[lane], states[lane]);
	}
#endif
}

