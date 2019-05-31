#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optix_prime/optix_prime.h>
#include "putil/Buffer.h"
#include "putil/Preprocessor.h"

#include "structs.hpp"

#define trace(s) std::cout << __FILE__ << ":" << __LINE__ << " : " << s << std::endl

// Defined in rays.cu
extern "C" void buildRays(ODRay*, int, int, const optix::float3 &, const optix::Matrix4x4 &, int);

int _main(int, char *argv[])
{
	std::vector<optix::float3> vertices;
	std::vector<optix::int3> indices;

	throw std::exception("Fill vectors with model loading or whatnot");

	/// Raytracing related code

	// Raytracing viewport dimensions
	int raytrace_w = 1280, raytrace_h = 720;

	// Setup OptiX Prime
	bool callOnCuda;
	RTPcontext rtpContext;
	RTPbuffertype rtpBufferType;
	if (rtpContextCreate(RTP_CONTEXT_TYPE_CUDA, &rtpContext) == RTP_SUCCESS)
	{
		CHK_PRIME(rtpContext, rtpContextSetCudaDeviceNumbers(rtpContext, 0, NULL));
		trace("OptiX Prime context initiated with CUDA");
		rtpBufferType = RTP_BUFFER_TYPE_CUDA_LINEAR;
		callOnCuda = true;
	}
	else
	{
		rtpContextCreate(RTP_CONTEXT_TYPE_CPU, &rtpContext);
		trace("OptiX Prime context initiated with CPU");
		rtpBufferType = RTP_BUFFER_TYPE_HOST;
		callOnCuda = false;
	}

	// Create buffer descriptors
	RTPbufferdesc rtpVerticesDesc, rtpIndicesDesc;
	CHK_PRIME(rtpContext, rtpBufferDescCreate(rtpContext, RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_HOST, vertices.data(), &rtpVerticesDesc));
	CHK_PRIME(rtpContext, rtpBufferDescSetRange(rtpVerticesDesc, 0, vertices.size()));
	CHK_PRIME(rtpContext, rtpBufferDescCreate(rtpContext, RTP_BUFFER_FORMAT_INDICES_INT3, RTP_BUFFER_TYPE_HOST, indices.data(), &rtpIndicesDesc));
	CHK_PRIME(rtpContext, rtpBufferDescSetRange(rtpIndicesDesc, 0, indices.size()));

	// Upload model for raytracing
	RTPmodel rtpModel;
	CHK_PRIME(rtpContext, rtpModelCreate(rtpContext, &rtpModel));
	CHK_PRIME(rtpContext, rtpModelSetTriangles(rtpModel, rtpIndicesDesc, rtpVerticesDesc));
	CHK_PRIME(rtpContext, rtpModelUpdate(rtpModel, 0));

	// Ray buffer
	RTPbufferdesc rtpRayDesc;
	Buffer<ODRay> raysBuffer(raytrace_w * raytrace_h, rtpBufferType, LOCKED);
	// Build rays
	optix::Matrix4x4 proj;
	optix::float3 origin;
	throw std::exception("Build a projection matrix of sorts");
	buildRays(raysBuffer.ptr(), raytrace_w, raytrace_h, origin, proj.inverse(), callOnCuda);

	CHK_PRIME(rtpContext, rtpBufferDescCreate(rtpContext, RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION, rtpBufferType, raysBuffer.ptr(), &rtpRayDesc));
	CHK_PRIME(rtpContext, rtpBufferDescSetRange(rtpRayDesc, 0, raysBuffer.count()));

	// Hit buffer
	RTPbufferdesc rtpHitDesc;
	Buffer<Hit> hitsBuffer(raytrace_w * raytrace_h, rtpBufferType, LOCKED);
	CHK_PRIME(rtpContext, rtpBufferDescCreate(rtpContext, RTP_BUFFER_FORMAT_HIT_T_TRIID_U_V, rtpBufferType, hitsBuffer.ptr(), &rtpHitDesc));
	CHK_PRIME(rtpContext, rtpBufferDescSetRange(rtpHitDesc, 0, hitsBuffer.count()));

	// Raytracing query
	RTPquery rtpQuery;
	CHK_PRIME(rtpContext, rtpQueryCreate(rtpModel, RTP_QUERY_TYPE_CLOSEST, &rtpQuery));
	CHK_PRIME(rtpContext, rtpQuerySetRays(rtpQuery, rtpRayDesc));
	CHK_PRIME(rtpContext, rtpQuerySetHits(rtpQuery, rtpHitDesc));
	CHK_PRIME(rtpContext, rtpQueryExecute(rtpQuery, 0));
    
	// Do things with the result
	Hit *data = hitsBuffer.hostPtr();

    // Cleanup
	rtpContextDestroy(rtpContext);
    
    return 0;
}

int main(int argc, char *argv[])
{
#ifdef NDEBUG
	trace("Release mode !");
#else
	trace("Debug mode !");
#endif
    try
    {
        return _main(argc, argv);
    }
    catch(std::exception &e)
    {
        std::cerr << e.what() << std::endl;
		return 0;
    }
}
