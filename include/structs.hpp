#ifndef INC_STRUCTS
#define INC_STRUCTS

#include <stdint.h>
#include <optixu/optixu_math_namespace.h>

struct ODRay
{
	optix::float3 origin;
	optix::float3 direction;
};
struct Hit
{
	float t;
	int triId;
	optix::float2 uv;
};

struct Color
{
	uint8_t r, g, b;
};

#endif