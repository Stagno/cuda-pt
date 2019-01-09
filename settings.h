#pragma once
#include <Windows.h>
#include <direct.h> 

struct Parameters {
	float3 cameraPosition;
	float3 cameraDirection;
	float3 cameraUp;
	float3 cameraRight;
	float n, f;
};

float ReadFloat(char* szSection, char* szKey, float fltDefaultValue)
{
	char szResult[255];
	char szDefault[255];
	float fltResult;
	sprintf(szDefault, "%f", fltDefaultValue);
	GetPrivateProfileString(szSection, szKey, szDefault, szResult, 255, scenePath);
	fltResult = atof(szResult);
	return fltResult;
}

float3 ReadFloat3(char* szSection, char* szKey, float3 fltDefaultValue)
{
	char szResult[255];
	char szDefault[255];
	float3 result;
	sprintf(szDefault, "(%f,%f,%f)", fltDefaultValue.x, fltDefaultValue.y, fltDefaultValue.z);
	GetPrivateProfileString(szSection, szKey, szDefault, szResult, 255, scenePath);
	sscanf(szResult, "(%f,%f,%f)", &result.x, &result.y, &result.z);
	return result;
}

int ReadMaterial(char* szSection, char* szKey)
{
	char szResult[255];
	char szDefault[255] = "lambert";
	int result = MAT_LAMBERT;
	GetPrivateProfileString(szSection, szKey, szDefault, szResult, 255, scenePath);
	
	if (strcmp(szResult, "lambert") == 0) result = MAT_LAMBERT;
	if (strcmp(szResult, "phong") == 0) result = MAT_PHONG;
	if (strcmp(szResult, "mirror") == 0) result = MAT_MIRROR;
	if (strcmp(szResult, "refractive") == 0) result = MAT_REFRACTIVE;
	return result;
}

void getSettingsFromFile(int &width, int &height, unsigned &max_num_paths, 
	unsigned &num_samples, float &scene_light_factor, float &p_absorption, 
	float &indexes_ratio, float &gammaCorrection, float &brightness) {

	char* buffer;

	// Get the current working directory:   
	if ((buffer = _getcwd(NULL, 0)) == NULL)
		perror("_getcwd error\n");
	else
	{
		printf("Working dir: %s\n", buffer);
		free(buffer);
	}

	width = GetPrivateProfileInt("RENDER_SETTINGS", "WIDTH", 512, scenePath);
	if (width%TILE_SIDE_LENGTH != 0) {
		fprintf(stderr, "width must be multiple of %d\n", TILE_SIDE_LENGTH);
		width = (width / TILE_SIDE_LENGTH);
		width *= TILE_SIDE_LENGTH;
	}
	height = GetPrivateProfileInt("RENDER_SETTINGS", "HEIGHT", 512, scenePath);
	if (height%TILE_SIDE_LENGTH != 0) {
		fprintf(stderr, "height must be multiple of %d\n", TILE_SIDE_LENGTH);
		height = (height / TILE_SIDE_LENGTH);
		height *= TILE_SIDE_LENGTH;
	}
	max_num_paths = GetPrivateProfileInt("RENDER_SETTINGS", "MAX_NUM_PATHS", 30, scenePath);
	num_samples = GetPrivateProfileInt("RENDER_SETTINGS", "NUM_SAMPLES", 1024, scenePath);
	scene_light_factor = ReadFloat("RENDER_SETTINGS", "SCENE_LIGHT_FACTOR", 0.5f);
	p_absorption = ReadFloat("RENDER_SETTINGS", "ABSORPTION_PROBABILITY", 0.25f);
	indexes_ratio = ReadFloat("RENDER_SETTINGS", "INDEXES_RATIO", 1.5f);
	gammaCorrection = 1.0f / ReadFloat("RENDER_SETTINGS", "GAMMA", 1.0f);
	brightness = ReadFloat("RENDER_SETTINGS", "BRIGHTNESS", 1.0f);
}

void getSceneFromFile(Sphere *spheres) {
	char section[255];
	for (int i = 0; i < N_SPHERES; i++)
	{
		sprintf(section, "SPHERE%d", i);
		spheres[i].diffuseColour = ReadFloat3(section, "diffuse", make_float3(0.5f, 0.5f, 0.5f));
		spheres[i].R = ReadFloat(section, "radius", 1.0f);
		spheres[i].C = ReadFloat3(section, "center", make_float3(0.0f, 0.0f, 0.0f));
		spheres[i].material = ReadMaterial(section, "material");
		spheres[i].emittedRadiance = ReadFloat3(section, "emissive", make_float3(0.5f, 0.5f, 0.5f));
	}

}

void setupCamera(Parameters &params, int width, int height) {
	params.cameraDirection = ReadFloat3("RENDER_SETTINGS", "CAM_DIRECTION", make_float3(0.0f,0.0f,1.0f)) + make_float3(float(width) / float(height), -1.0f, 0.0f);
	params.cameraPosition = ReadFloat3("RENDER_SETTINGS", "CAM_POSITION", make_float3(0.0f, 0.0f, -11.0f));
	params.cameraUp = ReadFloat3("RENDER_SETTINGS", "CAM_UP", make_float3(0.0f, 1.0f, 0.0f)) * 2.0f / (height);
	params.cameraRight = ReadFloat3("RENDER_SETTINGS", "CAM_RIGHT", make_float3(-1.0f, 0.0f, 0.0f)) * float(width) / float(height) * 2.0f / (width);
	params.n = 2.0f;
	params.f = 20.0f;
}