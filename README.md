# cuda-pt
NVIDIA CUDA implementation of a Quasi-Monte Carlo path tracing renderer. Vectors of the path are points from a Halton sequence, which is commonly used for faster convergence. 
Only very simple geometrical scenes can be rendered. 
The code traces the sampled paths in a full parallel way and it does not diverge.

Compile with
```
nvcc -o cuda-pt ray.cu
```
