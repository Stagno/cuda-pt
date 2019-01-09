#define SUPPRESS_ADDITIONAL_KERNELS
#include "cuda.h"
#include "book.h"
#include "cpu_bitmap.h"
#include "commons.h"
#include <cuda_runtime.h>
#include <random>
#include "init.h"
#include "rng.h"
#include "bmp.h"

//#define USE_TEXTURES

#define TILE_SIDE_LENGTH 256
#define SUBSAMPLES_SIDE_LENGTH 2
#define TILE_SIZE (TILE_SIDE_LENGTH * TILE_SIDE_LENGTH)
#define SUBSAMPLES (SUBSAMPLES_SIDE_LENGTH * SUBSAMPLES_SIDE_LENGTH)
#define INV_SS 1.0f / SUBSAMPLES
#define THREADS_PER_TILE TILE_SIZE * SUBSAMPLES

#define THREADS_PER_BLOCK  512
#define BLOCKS_PER_KERNEL (THREADS_PER_TILE / THREADS_PER_BLOCK)

#define SHARED_MEM_PER_BLOCK 16384
#define SHARED_MEM_PER_THREAD (SHARED_MEM_PER_BLOCK / THREADS_PER_BLOCK)

#define ISNORM(x) fabsf(length(x) - 1.0f) < 0.001f


#define MAX_TRACE_DEPTH     1.0e6f

#define MAT_LAMBERT 1
#define MAT_MIRROR 2
#define MAT_PHONG 3

char scenePath[255];

#include "Sphere.h"
#include "settings.h"

__constant__ aligned_t<Sphere, N_SPHERES> spheres;
#ifdef USE_TEXTURES
texture<uchar4, 2, cudaReadModeNormalizedFloat> tex;
#endif
/* Using shared memory (accessed in a strided way) to save on registers. */
template<block_size_t block_size>
struct ALIGN(16) shared_data_t {
	float radiances[block_size * 3]; // this array must occupy a multiple of 16 bytes, if less => padding
	const float *ptrToRands[block_size]; // pointers to the first randoms // this must occupy a multiple of 16 bytes, if less => padding

};

// Traces ray until an object is intersected or max trace depth is reached, returns id of object (otherwise -1)
//and saves hit point in *hitPoint
// Direction must be normalized.
inline __device__ int trace(float3 startPoint, float3 direction, float3 &hitPoint)
{
	float t, min_t = MAX_TRACE_DEPTH;
	int min_i;

	// Search is along ray direction. Hits are identified by coordinate t on the ray line.
	
	for (int i = 0; i < N_SPHERES; i++) {
		t = (spheres.index(i))->intersect(startPoint, direction);
		if (t < min_t) {
			min_i = i;
			min_t = t;
		}
	}
	if (min_t == MAX_TRACE_DEPTH) return -1; // NO HIT
	hitPoint = startPoint + min_t * direction;
	return min_i;
}

/*VARIANT: works for pseudo randoms*/
inline __device__ float3 importanceSampleHemisphereVariant(const float *ptrToRand, 
															float3 yAxis, unsigned i)
{
	
	float phi = *(ptrToRand + i * THREADS_PER_TILE) * PI * 2.0f;
	float thetha = acosf(sqrtf(*(ptrToRand + (i + 1) * THREADS_PER_TILE)));
	float sinThetha = sinf(thetha);
	if (yAxis.x == 0.0f && yAxis.z == 0.0f)
		return make_float3(sinThetha * cosf(phi), cosf(thetha), sinThetha * sinf(phi));
	float3 zAxis = normalize(cross(make_float3(0.0f, 1.0f, 0.0f), yAxis));
	float3 xAxis = normalize(cross(yAxis, zAxis));
	float3 sample = xAxis * sinThetha * cosf(phi) + yAxis * cosf(thetha) + zAxis * sinThetha * sinf(phi);

	return sample;
}

/* Sample hemisphere by just using the precalculated halton point from the sequence and rotate it to the normal basis*/
inline __device__ float3 HaltonSampleHemisphere(const float *ptrToRand,
	float3 yAxis, unsigned i)
{
	
	float3 zAxis = normalize(cross(make_float3(0.0f, 1.0f, 0.0f), yAxis)); // trick to have a zAxis
	float3 xAxis = normalize(cross(yAxis, zAxis)); // xAxis calculated as cross prod. of y and z
	float3 sample = xAxis * *(ptrToRand + i * THREADS_PER_TILE) +
					zAxis * *(ptrToRand + (i + 1) * THREADS_PER_TILE) +
					yAxis * *(ptrToRand + (i + 2) * THREADS_PER_TILE); // build vector with components
	return sample;
}
// pdf =  cos(thetha) / pi
 __device__ float lambertPdf(float cos) {
	return cos / PI;
}
 // pdf = sin(thetha) * cos(thetha) / pi
 inline __device__ float lambertPdfVariant(float cos) {
	return (cos*sqrtf(1 - cos * cos)) / PI;
}


/* Path tracing function. Returns estimated radiance. inDir must be normalized.*/
 inline __device__ float3 path_trace(const float *ptrToRand, const float absorption_factor, const float indexes_ratio,
								int obj, float3 startPoint, float3 inDir, unsigned num_paths)
{
	float3 normal, outDir;
	float cosThetha;
	// estimate of radiance before any reflection, it is given by the emittedRadiance of the sphere hit
	float3 estimatedRadiance = spheres.index(obj)->emittedRadiance;
	float3 factor = make_float3(1.0f, 1.0f, 1.0f); // factor that multiplies the following part of the radiance to be calc.

	unsigned i = 0;
	while (i<num_paths) // Russian roulette
	{
		normal = spheres.index(obj)->normal(startPoint); // get normal at hit point
		// for each different material there is a different brdf
		switch (spheres.index(obj)->material) {
		case MAT_MIRROR:
			/* case Mirror material */
			outDir = reflect(inDir, normal); // brdf is a dirac delta on mirror direction
			break;
		case MAT_LAMBERT:
			/* case Lambert material */
			outDir = HaltonSampleHemisphere(ptrToRand, normal, 3 * i); // sample a direction on the normal hemisphere 

			cosThetha = dot(normal, outDir); // thetha is the angle between output dir. and normal dir.
#ifdef USE_TEXTURES
			float2 uv = spheres.index(obj)->getUV(normal); // gets the UV corresponding to the normal dir.
			int u = uv.x * 1023.0f, v = uv.y * 511.0f; // scales the UV by the size of the texture (1024x512)
			float3 texel = make_float3(tex2D(tex, u, v)); // gets the texel at coord. (u,v)
			factor *= texel * INV_PI  // factor = BRDF * cos  / pdf
				* clamp(dot(outDir, normal), 0.0f, 1.0f)
				/ lambertPdf(cosThetha);
#else
			factor *= spheres.index(obj)->diffuseColour * INV_PI  // factor = BRDF * cos  / pdf
				* clamp(dot(outDir, normal), 0.0f, 1.0f)
				/ lambertPdf(cosThetha);
#endif
			break;
		case MAT_REFRACTIVE:
			/* case Refractive material */
			/* This part will calculate the incident ray from the transmitted ray of the refraction from inside material to air.
				See notes.
				*/
			outDir = -inDir; // to save on regs i'll just use outDir as temp var, first it is set as the opposite of the incoming ray
			float cosThetha_t = dot(normal, outDir); // cosine of angle between normal and (opposite of) incoming ray
			float squaredSinThetha_t = 1 - cosThetha_t * cosThetha_t; // sin^2 (thetha)
			float3 tang = (outDir - normal * cosThetha_t) * rsqrtf(squaredSinThetha_t); // tangent vector
			float squaredCosThetha_i = 1 - indexes_ratio * indexes_ratio * squaredSinThetha_t;
			outDir = -(normal * sqrtf(squaredCosThetha_i) + tang * sqrtf(1 - squaredCosThetha_i)); // outgoing vector
			
			// Calculate starting point of internal ray (see notes, triangle inside sphere)
			startPoint -= dot(outDir, normal) * spheres.index(obj)->R * 2.0f * outDir;
			
			// This part calculates the incident ray from the transmitted ray of the refraction from air to inside material.
			normal = - spheres.index(obj)->normal(startPoint); // normal at the starting point
			outDir = -outDir; // take opposite of ray
			cosThetha_t = dot(normal, outDir);
			squaredSinThetha_t = 1 - cosThetha_t * cosThetha_t;
			tang = (outDir - normal * cosThetha_t) * rsqrtf(squaredSinThetha_t);
			squaredCosThetha_i = 1 - 1.0f / (indexes_ratio * indexes_ratio) * squaredSinThetha_t;
			outDir = -(normal * sqrtf(squaredCosThetha_i) + tang * sqrtf(1 - squaredCosThetha_i)); // outgoing vector

			
			break;
		default:
			break;
		}
		
		obj = trace(startPoint, outDir, startPoint); // traces next point
		
		inDir = outDir; // incoming direction for the new point is the outgoing direction for the old point

		if (obj != -1) {
			estimatedRadiance += factor * spheres.index(obj)->emittedRadiance; // weigh the emitted radiance of the current
																				// sphere by the factor obtained at this point
		}
		else
			break;
		i++;
	}
	
	return clamp(estimatedRadiance  * absorption_factor, 0.0f, 1.0f); // apply absorption factor and saturate to 1
}

/* Generates perspective ray direction. xs and ys are global screen coordinate, normalization is in camera parameters */
 inline __device__ float3 perspectiveRay(const Parameters &p, float2 pos)
{
	return normalize(p.cameraDirection + pos.x * p.cameraRight + pos.y * p.cameraUp);
}

 /* Do the average and bring to RGB, save to frame buffer */
 __global__ void NormalizeAndConvertToRGB(const uint32_t spp, float *radiance, unsigned char *frame, float gammaCorrection, float brightness)
 {
	 int id = threadIdx.x + blockIdx.x * blockDim.x;
	 frame[id * 4 + 0] = (int)(powf(clamp(brightness * radiance[id * 3 + 0] / float(spp), 0.0f, 1.0f),gammaCorrection) * 255.0f);
	 frame[id * 4 + 1] = (int)(powf(clamp(brightness * radiance[id * 3 + 1] / float(spp), 0.0f, 1.0f),gammaCorrection) * 255.0f);
	 frame[id * 4 + 2] = (int)(powf(clamp(brightness * radiance[id * 3 + 2] / float(spp), 0.0f, 1.0f),gammaCorrection) * 255.0f);
	 frame[id * 4 + 3] = 255;
 }
 
 /* Takes 4 subsamples of a pixel of tile with coords tileCoords, by estimating its radiance. Then merges the subsamples. Saves in output buffer*/
__global__ void sampleTile( float * const output,  
							const Parameters p,
							const float * const rands,
							const uint2 tileCoords,
							const uint32_t num_paths,
							const uint2 imageRes,
							const float absorption_factor,
							const float indexes_ratio
							) 
{

	__shared__ shared_data_t<THREADS_PER_BLOCK> shared_data; // shared mem to save on registers

    // id of thread (there are THREADS_PER_TILE threads to render a tile)
	uint32_t id = __umul24(THREADS_PER_BLOCK, blockIdx.x) + threadIdx.x;
	uint32_t ssx = id & 1; // take least significant bit => x "coordinate" inside pixel for subsampling
	uint32_t ssy = (id & 2) >> 1; // take second least significant bit => y "coordinate" inside pixel for subsampling
	shared_data.ptrToRands[threadIdx.x] = rands + id; // pointer to first random
	id >>= 2; // now id identifies a pixel in the tile, let's calculate x and y coordinates
	uint32_t x = id % TILE_SIDE_LENGTH; // x is a coordinate for the pixel in the tile (least sign. bits of id)
	uint32_t y = id / TILE_SIDE_LENGTH; // y is a coordinate for the pixel in the tile (most sign. bits of id)
	float2 coords = make_float2(x, y) +make_float2(tileCoords);  // global coords. of pixel
	float2 ss = make_float2(ssx, ssy); // subpixel coords. in a vector
	
	float2 subrands = make_float2(*(shared_data.ptrToRands[threadIdx.x]), 
		*(shared_data.ptrToRands[threadIdx.x] + THREADS_PER_TILE));//first 2 randoms consumed, global mem is strided accessed
	// subpixel position is choosen at random inside the pixel
	float2 pos = (ss + subrands)*make_float2(0.5f, 0.5f) + coords; // global coordinates of subpixel

	// calc. primary ray, some computations has already been done when loading camera param.
	float3 primaryRay = perspectiveRay(p, pos); 
	

	float3 hitPoint;
	// trace first point hit
	int hitSphere = trace(p.cameraPosition, primaryRay, hitPoint);

	// radiance samples are stored in shared mem. to later merge them
	strided_mem_t<THREADS_PER_BLOCK, float> radiance(shared_data.radiances, threadIdx.x);
	if (hitSphere == -1) { // no hit, black sample
		radiance.set(0, 0.0f);
		radiance.set(1, 0.0f);
		radiance.set(2, 0.0f);
		return;
	}
	
	Sphere *sphere = (spheres.index(hitSphere)); // hit sphere id
	if (hitSphere != -1) { // if there is a hit
		// sample radiance by path tracing, the number of "paths" to follow is fixed and precalculated (sampled from geometric distr.)
		radiance = path_trace(shared_data.ptrToRands[threadIdx.x] + 2 * THREADS_PER_TILE, absorption_factor, indexes_ratio,
							hitSphere, hitPoint, primaryRay, num_paths);
		__syncthreads();
		
	}
	

	// Merging subsamples

	__syncthreads();

	if ((threadIdx.x & (SUBSAMPLES - 1)) == 0) {
		
		strided_mem_t<THREADS_PER_BLOCK, float> radiance1(shared_data.radiances, threadIdx.x+1);
		strided_mem_t<THREADS_PER_BLOCK, float> radiance2(shared_data.radiances, threadIdx.x+2);
		strided_mem_t<THREADS_PER_BLOCK, float> radiance3(shared_data.radiances, threadIdx.x+3);
		//	average subsamples
		float3 final_radiance = make_float3(radiance.r() + radiance1.r() + radiance2.r() + radiance3.r(),
											radiance.g() + radiance1.g() + radiance2.g() + radiance3.g(),
											radiance.b() + radiance1.b() + radiance2.b() + radiance3.b()
											) * INV_SS;
		// save in output
		uint32_t index = (tileCoords.x + x + imageRes.x * (tileCoords.y + y));
		output[index * 3 + 0] += final_radiance.x;
		output[index * 3 + 1] += final_radiance.y;
		output[index * 3 + 2] += final_radiance.z;
		
	}
}



// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};


/* renders the image by executing kernels on the gpu */
void renderTiles(cudaStream_t stream, unsigned char *frame, 
				std::geometric_distribution<> &geom_dist, std::mt19937 &mt_rng,
				Parameters &parameters, int width, int height, 
				const unsigned int max_num_paths, const unsigned int num_samples,
				const float p_absorption, const float indexes_ratio, const float gammaCorrection, const float brightness) {

	
	const unsigned int num_threads = THREADS_PER_TILE; // need (n. of pixels of tile times n. of subsamples) threads
	unsigned int num_paths, randoms_per_thread, want;
	float pow_p_abs, absorption_factor;

	float *d_randoms;
	// Malloc enough space on GPU for the baked randoms used by 1 sampleTile call: for each thread need 2 randoms to 
	// randomize subpixel position and 3 randoms per path (the xyz from halton sampling)
	HANDLE_ERROR(cudaMalloc((void**)&d_randoms,
		(2 + 3 * max_num_paths) * num_threads * sizeof(float)));

	// Buffer for the output radiance from sampleTile call
	float *radiance;
	HANDLE_ERROR(cudaMalloc((void**)&radiance,
		3 * width * height * sizeof(float)));
	// Zeroes the buffer
	cudaMemset(radiance, 0, sizeof(float) * width * height * 3);

	// Start performance measure
	cudaEvent_t     start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, stream));

	int num_tiles_x = width / TILE_SIDE_LENGTH, num_tiles_y = height / TILE_SIDE_LENGTH; // Expecting width and height to be multiples of TILE_SIDE_LENGTH
	uint32_t subsequence = 0; // subsequence index, needed to respect the halton sequence order across multiple samples
	for (unsigned i = 0; i < num_samples; i++) { // Call the rendering kernels num_samples times to get desired number of spp
		/* num_paths is sampled from a geometric distribution */
		/* ATTENZIONE: la media di questa geometrica è (1-p)/p */
		num_paths = geom_dist(mt_rng);
		if (num_paths > max_num_paths) num_paths = max_num_paths; // cut the tail of the distribution
		// precalc. factor to apply to sampled radiance to weight it by the probability of reaching num_paths paths
		pow_p_abs = pow(1 - p_absorption, num_paths);
		absorption_factor = 1.0f / (pow_p_abs  * p_absorption); 
		// bake the required number of quasi randoms and place them on the global mem to be further accessed by sampleTile
		bake_halton<<<BLOCKS_PER_KERNEL, THREADS_PER_BLOCK, 0, stream >>>(3, num_samples, subsequence, num_paths, d_randoms);
		
		for (int x = 0; x < num_tiles_x; x++) // iterate horizontally across tiles
			for (int y = 0; y < num_tiles_y; y++) // iterate vertically across tiles
				sampleTile<<<BLOCKS_PER_KERNEL,THREADS_PER_BLOCK,0,stream>>>(
					radiance, parameters, d_randoms, make_uint2(x , y) * TILE_SIDE_LENGTH, 
					num_paths, make_uint2(width,height), absorption_factor, indexes_ratio); // takes 4 (sub)samples 
																							// for each pixel of the (x,y) tile and 
																							// averages them, adds the result
																							// to the radiance buffer
		subsequence ++; 
	}
	// do the average of samples for each pixel in the image and converts from radiance ([0,1]^3) to RGB, saves output in frame buffer
	NormalizeAndConvertToRGB<<<width,height,0,stream>>>(num_samples, radiance, frame, gammaCorrection, brightness);
	// stops performance measurement
	HANDLE_ERROR(cudaEventRecord(stop, stream));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	float   elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
		start, stop));
	printf("Time to generate:  %3.1f ms\n", elapsedTime);
	// clean up
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	
	HANDLE_ERROR(cudaFree(radiance));

	cudaFree(d_randoms);
	
}

int main( void ) {

    DataBlock   data;
	int width, height;
	unsigned int max_num_paths, num_samples;
	float scene_light_factor, p_absorption, indexes_ratio, gammaCorrection, brightness;

	//	Load settings from file
	char name[255];
	printf("Enter scene file name: ");
	scanf("%s", name);
	strcpy(scenePath, "./");
	strcat(scenePath, name);
	getSettingsFromFile(width, height, max_num_paths, num_samples, scene_light_factor, p_absorption, indexes_ratio, gammaCorrection, brightness);
	
	// CPUBitmap class used just to display a bitmap coming from GPU calculations
	CPUBitmap bitmap(width, height, &data);
	unsigned char   *dev_frame, *dev_texture;

	//	Display useful info about the main GPU device
	cudaDeviceProp prop;
	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	if (count == 0) {
		fprintf(stderr, "ERROR: no cuda device found");
		return 1;
	}
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
	printf("Multiprocessor count: %d\n",
		prop.multiProcessorCount);
	printf("Registers per block: %d\n",
		prop.regsPerBlock);


#ifdef USE_TEXTURES
	//	Load texture to GPU and bind
	int texWidth, texHeight;
	long texSize;
	BYTE *texBuffer = ConvertBMPToRGBBuffer(LoadBMP(&texWidth, &texHeight, &texSize, "./texture.bmp"), texWidth, texHeight);
	HANDLE_ERROR(cudaMalloc((void**)&dev_texture, 4 * texWidth * texHeight));
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
	HANDLE_ERROR(cudaBindTexture2D(NULL, tex, dev_texture, desc, texWidth, texHeight, texWidth*sizeof(uchar4)));
	HANDLE_ERROR(cudaMemcpy(dev_texture, texBuffer, 4 * texWidth * texHeight, cudaMemcpyHostToDevice));
#endif
	// allocate memory on the GPU for the output bitmap
	HANDLE_ERROR(cudaMalloc((void**)&dev_frame,
		bitmap.image_size()));

	Sphere temp_spheres[N_SPHERES];
	Parameters params;
	//	Load spheres from file
	getSceneFromFile(temp_spheres);
	INIT_DATA; // applies scene light factor to diffusive colours and precalc. squares of radii
	setupCamera(params,width,height); // precalc. vectors for the camera

	// upload sphere data to GPU
	HANDLE_ERROR(cudaMemcpyToSymbol(spheres, temp_spheres,
		sizeof(Sphere) * N_SPHERES));
	
	// using a single stream for 2 kernels alternating
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	// geometric distribution setup
	std::random_device rd;
	std::mt19937 gen(rd());
	std::geometric_distribution<> geom_dist(p_absorption);
	// render the image
	renderTiles(stream, dev_frame, geom_dist, gen, params, width, height, 
		max_num_paths, num_samples, p_absorption, indexes_ratio, gammaCorrection, brightness);

	// copy bitmap back from the GPU for display
	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_frame,
		bitmap.image_size(),
		cudaMemcpyDeviceToHost));

	//clean up
	HANDLE_ERROR(cudaFree(dev_frame));
#ifdef USE_TEXTURES
	cudaUnbindTexture(tex);
	cudaFree(dev_texture);
	free(texBuffer);
#endif
	cudaStreamDestroy(stream);

	// display
	bitmap.display_and_exit();
	system("pause");
	return 0;
	
}

