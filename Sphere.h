#pragma once


#define MAT_LAMBERT 1
#define MAT_MIRROR 2
#define MAT_PHONG 3
#define MAT_REFRACTIVE 4

struct Sphere {
	float3  diffuseColour;
	float   R, Rsquared;
	float3  C;
	int		material;
	float3 emittedRadiance;
	char pad[64 - 48]; // padding added to align on memory

	/* Returnes normal direction (normalized). P must be a point on the surface of the sphere!!*/
	inline __device__ float3 normal(float3 P) {
		return (P - C) / R;
	}

	// Calculate UV coords. from normal
	inline __device__ float2 getUV(float3 normal) {
		float3 up = make_float3(0.0f, 1.0f, 0.0f); // Vertical direction (pos. y)
		float v = (dot(-up, normal) + 1.0f) / 2.0f; // v coord.
		float3 proj = normalize(make_float3(normal.x, 0.0f, normal.z)); // proj. of normal on xz-plane
		float index = dot(proj, make_float3(0.0f, 0.0f, 1.0f)); // wheter proj. has pos. z or neg.
		float u = (index > 0.0f ? (dot(proj, make_float3(-1.0f, 0.0f, 0.0f)) + 1.0f) : (dot(proj, make_float3(1.0f, 0.0f, 0.0f)) + 3.0f)) / 4.0f; // u coord.
		return make_float2(u, v);
	}

	// Ray-sphere intersection. See notes.
	inline __device__ float intersect(float3 P, float3 d) {


		float a = dot(d, d);
		float b = sumElements(d * (P - C) * 2.0f);
		float c = dot(C, C) + dot(P, P) - 2.0f * dot(C, P) - Rsquared;

		float delta = b*b - 4.0f * a * c;
		if (delta < 0.0f) return MAX_TRACE_DEPTH;

		float t = (-b - sqrtf(delta)) / (2.0f * a);
		if (t < 0.0f) return MAX_TRACE_DEPTH;

		return t;

	}

};

/*

struct Plane {
int type;
float coord;
float minU, minV, maxU, maxV;
float facing; // +1 if normal is along positive axis, -1 otherwise
float3  diffuseColour;
int		material;
float3 emittedRadiance;

__device__ float3 normal(float3 P) {
switch (type) {
case XPLANE:
return facing * make_float3(1.0f, 0.0f, 0.0f);
case YPLANE:
return facing * make_float3(0.0f, 1.0f, 0.0f);
case ZPLANE:
return facing * make_float3(0.0f, 0.0f, 1.0f);
}
}

__device__ float intersect(float3 P, float3 d) {
float t,u,v;
switch (type) {
case XPLANE:
t = (coord - P.x) / d.x;
if (t < 0.0f || d.x * facing > 0) return MAX_TRACE_DEPTH;
u = P.y + t * d.y;
v = P.z + t * d.z;
break;
case YPLANE:
t = (coord - P.y) / d.y;
if (t < 0.0f || d.y * facing > 0) return MAX_TRACE_DEPTH;
u = P.x + t * d.x;
v = P.z + t * d.z;
break;
case ZPLANE:
t = (coord - P.z) / d.z;
if (t < 0.0f || d.z * facing > 0) return MAX_TRACE_DEPTH;
u = P.x + t * d.x;
v = P.y + t * d.y;
break;
}

if (u<minU || u>maxU || v<minV || v>maxV) return MAX_TRACE_DEPTH;
return t;
}
};
*/