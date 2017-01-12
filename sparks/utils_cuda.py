import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.tools
import pycuda.autoinit

THREADS_PER_BLOCK = 256

module = SourceModule("""

    __device__ uchar3 convert_one_pixel_to_hsv(unsigned char* pixel) 
    {
    float r, g, b;
    float h, s, v;
    
    r = pixel[0] / 255.0f;
    g = pixel[1] / 255.0f;
    b = pixel[2] / 255.0f;
    
    float max = fmax(r, fmax(g, b));
    float min = fmin(r, fmin(g, b));
    float diff = max - min;
    
    v = max;
    
    if(v == 0.0f) { // black
        h = s = 0.0f;
    } else {
        s = diff / v;
        if(diff < 0.001f) { // grey
            h = 0.0f;
        } else { // color
            if(max == r) {
                h = 60.0f * (g - b)/diff;
                if(h < 0.0f) { h += 360.0f; }
            } else if(max == g) {
                h = 60.0f * (2 + (b - r)/diff);
            } else {
                h = 60.0f * (4 + (r - g)/diff);
            }
        }       
    }
    return make_uchar3(h/2, s*255, v*255);
    }

    __global__ void process(unsigned char* buffer, long size, float *histo)
    {
        __shared__ unsigned int temp[256][2];

        temp[threadIdx.x][0] = 0;
        temp[threadIdx.x][1] = 0;
        __syncthreads();

        int i = threadIdx.x * 4 + blockIdx.x * blockDim.x * 4;
        int offset = blockDim.x * gridDim.x * 4;

        while (i < size) {
            uchar3 hsv = convert_one_pixel_to_hsv(&(buffer[i]));
            atomicAdd( &temp[hsv.x][0], 1);
            atomicAdd( &temp[hsv.y][1], 1);            
            i += offset;
        }
        __syncthreads();

        atomicAdd( &(histo[threadIdx.x*2]), temp[threadIdx.x][0]);
        atomicAdd( &(histo[threadIdx.x*2 + 1]), temp[threadIdx.x][1]);
    }
    """)

process = module.get_function("process")

def gpu_process(image, histogram):
    process(cuda.In(image), np.int32(image.size / 4), cuda.InOut(histogram),
            block=(THREADS_PER_BLOCK, 1, 1),
            grid=(10, 1))
    return histogram