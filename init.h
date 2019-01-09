#pragma once

#define N_SPHERES 12

#define INIT_DATA \
for (int j = 0; j < N_SPHERES; j++) { \
temp_spheres[j].diffuseColour *= scene_light_factor;\
temp_spheres[j].Rsquared = temp_spheres[j].R * temp_spheres[j].R; \
}
