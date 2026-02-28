# Cuda based particle simulation

Just a simple particle simulation using Cuda, as I wanted to work on my C++ skills and learn something new. Still a work in progress.

## Compile using this so the compiler stops complaining about it not finding `cuda_gl_interop.h`
```bash
$ nvcc src/main.cpp src/sim.cu \
    -I./src \
    -I/usr/local/cuda/targets/x86_64-linux/include \
    -L/usr/local/cuda/targets/x86_64-linux/lib \
    -lglfw -lGLEW -lGL -lcuda -lcudart \
    -o particles
$ ./build/particles
```
