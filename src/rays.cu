#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <optix_prime/optix_prime.h>
#include <optixu/optixu_matrix_namespace.h>

#include "structs.hpp"

using namespace optix;

__host__ __device__ float3 makeTarget(int x, int y, int w, int h, const Matrix4x4 &invProj)
{
	float4 r = invProj * make_float4(float(x) * 2 / w - 1, float(y) * 2 / h - 1, 0.5, 1);
	return make_float3(r / r.w);
}

__global__ void buildRaysKernel(ODRay *ptr, int w, int h, const float3 origin, const Matrix4x4 invProj)
{
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= w * h)
		return;
	ODRay r;
	r.origin = origin;
	r.direction = normalize(makeTarget(tid % w, tid / w, w, h, invProj) - origin);
	ptr[tid] = r;
}

extern "C" void buildRays(ODRay *ptr, int w, int h, const float3 &origin, const Matrix4x4 &invProj, int callOnDevice)
{
	if (callOnDevice)
	{
		int threads = w * h;
		buildRaysKernel<<<(threads + 1023) / 1024, 1024>>>(ptr, w, h, origin, invProj);
	}
	else
	{
		ODRay r;
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				uint tid = i + j * w;
				r.origin = origin;
				r.direction = normalize(makeTarget(i, j, w, h, invProj) - origin);
				ptr[tid] = r;
			}
		}
	}
}
