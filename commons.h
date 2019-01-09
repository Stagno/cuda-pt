#ifndef HELPER_TRACER
#define HELPER_TRACER
#include "helper_math.h"


#define PI 3.14159265359f
#define INV_PI 0.318309886f

#define ALIGN(x)		__declspec(align(x))
#define UNIVERSAL inline __device__ __host__

typedef unsigned char uchar_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

//typedef uint16_t bool_t;
typedef uint32_t bool_t;

typedef uint16_t block_size_t;


inline __host__ __device__ float sumElements(float3 v)
{
	return v.x + v.y + v.z;
}

template<uint16_t block_size, typename T>
struct strided_mem_t {

	T * const p;
	unsigned t;

	UNIVERSAL strided_mem_t(T *q, unsigned thread) : p(q), t(thread) {}
	UNIVERSAL T get(const unsigned idx) const { return p[block_size*idx + t]; }
	UNIVERSAL void set(const unsigned idx, const T rhs) const { p[block_size*idx + t] = rhs; }

	UNIVERSAL T x() const { return get(0); }
	UNIVERSAL T y() const { return get(1); }
	UNIVERSAL T z() const { return get(2); }

	UNIVERSAL T r() const { return get(0); }
	UNIVERSAL T g() const { return get(1); }
	UNIVERSAL T b() const { return get(2); }

	UNIVERSAL void operator= (float3 rvalue) { set(0, rvalue.x); set(1, rvalue.y); set(2, rvalue.z); }
	UNIVERSAL void operator+= (float3 rvalue) {
		p[t] += rvalue.x; p[block_size * 1 + t] += rvalue.y; p[block_size * 2 + t] += rvalue.z;
	}
	UNIVERSAL void operator/= (float rvalue) {
		p[t] /= rvalue; p[block_size * 1 + t] /= rvalue; p[block_size * 2 + t] /= rvalue;
	}
};

template<typename T, size_t num>
struct aligned_t {

	char ALIGN(16) raw[sizeof(T)*num];

	__device__ T *index(const unsigned i) { return reinterpret_cast<T*>(raw) + i; }
	__device__ const T *index(const unsigned i) const { return reinterpret_cast<const T*>(raw) + i; }


};

#endif