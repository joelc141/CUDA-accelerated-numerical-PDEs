# CUDA-accelerated-numerical-PDEs
CUDA project(still working on it)
Working on using PDEs to accelerate processing of PDEs. Some PDEs have many dimensions or variables and can become incredibly expensive to calculate. To increase efficiency we can write them as repeated matrix multiplication and use CUDA's linear algebra package ontop of this. This will work for higher dimensional PDEs, but you will need to stack the matrices in an array before sending them to the GPU, to prevent callback to the CPU.


The other code is just a simple PDE in cuda, this isnt very practical but serves as a start for understanding basic forloops. This one was provided by Kevin Cooper in a GPU computing class.
