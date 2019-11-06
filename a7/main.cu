#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
extern "C" { 
    #include "libs/bitmap.h"
}

namespace cg = cooperative_groups;

// BLOCKX * BLOCKY can max be 1024 (threads)
// GRIDX * GRIDY can max be 28 (MSs)
#define BLOCKY  32
#define BLOCKX  32
#define GRIDX   7
#define GRIDY   4 

#define ERROR_EXIT -1
#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// Constant memory for GPU
__constant__ int constFilterGPU[25];

// Convolutional Filter Examples, each with dimension 3,
// gaussian filter with dimension 5
// If you apply another filter, remember not only to exchange
// the filter but also the filterFactor and the correct dimension.

/*
int const sobelYFilter[] = {-1, -2, -1,
                            0, 0, 0,
                            1, 2, 1};
float const sobelYFilterFactor = (float)1.0;

int const sobelXFilter[] = {-1, -0, -1,
                            -2, 0, -2,
                            -1, 0, -1, 0};
float const sobelXFilterFactor = (float)1.0;
*/

int const laplacian1Filter[] = {-1, -4, -1,
                                -4, 20, -4,
                                -1, -4, -1};

float const laplacian1FilterFactor = (float)1.0;

/*
int const laplacian2Filter[] = {0, 1, 0,
                                1, -4, 1,
                                0, 1, 0};
float const laplacian2FilterFactor = (float)1.0;

int const laplacian3Filter[] = {-1, -1, -1,
                                -1, 8, -1,
                                -1, -1, -1};
float const laplacian3FilterFactor = (float)1.0;

// Bonus Filter:
int const gaussianFilter[] = {1, 4, 6, 4, 1,
                              4, 16, 24, 16, 4,
                              6, 24, 36, 24, 6,
                              4, 16, 24, 16, 4,
                              1, 4, 6, 4, 1};
float const gaussianFilterFactor = (float)1.0 / 256.0;
*/

// CPU serial - Apply convolutional filter on image data
void applyFilter(unsigned char **out, unsigned char **in, const int width, const int height, 
                 const int *filter, const int filterDim, const float filterFactor)
{
    const int filterCenter = filterDim / 2;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int aggregate = 0;
            for (int ky = 0; ky < filterDim; ky++)
            {
                int nky = filterDim - 1 - ky;
                for (int kx = 0; kx < filterDim; kx++)
                {
                    int nkx = filterDim - 1 - kx;

                    int yy = y + (ky - filterCenter);
                    int xx = x + (kx - filterCenter);
                    if (xx >= 0 && xx < width && yy >= 0 && yy < height)
                        aggregate += in[yy][xx] * filter[nky * filterDim + nkx];
                }
            }
            aggregate *= filterFactor;
            if (aggregate > 0)
                out[y][x] = (aggregate > 255) ? 255 : aggregate;
            else
                out[y][x] = 0;
        }
    }
}

// GPU basic - Apply convolutional filter on image data
__global__ void device_applyFilter(unsigned char *out, const unsigned char *in, const int width, const int height, 
                                   const int *filter, const int filterDim, const float filterFactor)
{
    const int x = blockIdx.x * BLOCKX + threadIdx.x;
    const int y = blockIdx.y * BLOCKY + threadIdx.y;
    if (x < width && y < height)
    {
        const int filterCenter = filterDim / 2;
        int aggregate = 0;
        for (int ky = 0; ky < filterDim; ky++)
        {
            int nky = filterDim - 1 - ky;
            for (int kx = 0; kx < filterDim; kx++)
            {
                int nkx = filterDim - 1 - kx;
                int yy = y + (ky - filterCenter);
                int xx = x + (kx - filterCenter);
                if (xx >= 0 && xx < width && yy >= 0 && yy < height)
                    aggregate += in[xx + yy*width] * filter[nky * filterDim + nkx];
            }
        }
        aggregate *= filterFactor;
        if (aggregate > 0)
            out[x + y*width] = (aggregate > 255) ? 255 : aggregate;
        else
            out[x + y*width] = 0;
    }
}

// GPU shared memory - Apply convolutional filter on image data
__global__ void device_applyFilter_sm(unsigned char *out, const unsigned char *in, const int width, const int height, const int N, 
                                      const int *filter, const int filterDim, const float filterFactor)
{
    extern __shared__ unsigned char data_sm[];
    int *filter_sm = (int*)&data_sm[N];
    const int x = blockIdx.x * BLOCKX + threadIdx.x;
    const int y = blockIdx.y * BLOCKY + threadIdx.y;
    const int xLocal = threadIdx.x;
    const int yLocal = threadIdx.y;
    const int filterCenter = (filterDim / 2);

    if(xLocal == 1 && yLocal == 1)
    {
        for (int i = 0; i < filterDim*filterDim; i++)
            filter_sm[i] = filter[i];
    }

    if (x < width && y < height)
    {
        // Tranfer center to shared memory
        data_sm[(xLocal+filterCenter) + (yLocal+filterCenter)*(BLOCKX+filterCenter*2)] = in[x + y*width];
        
        // Transfer west border to shared memory
        if(threadIdx.x == 0)
        {
            for(int i = 1; i <= filterCenter; i++)
            {
                if(threadIdx.y == 0) // North West corner
                {
                    for(int j = 0; j <= filterCenter; j++)
                    {
                        if(x - i >= 0 && y - j >= 0)
                            data_sm[(xLocal+filterCenter-i) + (yLocal+filterCenter-j)*(BLOCKX+filterCenter*2)] = in[x-i + (y-j)*width];
                    }
                }
                else
                {
                    if(x - i >= 0)
                        data_sm[(xLocal+filterCenter-i) + (yLocal+filterCenter)*(BLOCKX+filterCenter*2)] = in[x-i + y*width];
                }
            }
        }
        
        // Transfer north border to shared memory
        if(threadIdx.y == 0)
        {
            for(int i = 1; i <= filterCenter; i++)
            {
                if(threadIdx.x == BLOCKX-1) // North East corner
                {
                    for(int j = 0; j <= filterCenter; j++)
                    {
                        if(x + j < width && y - i >= 0)
                            data_sm[(xLocal+filterCenter+j) + (yLocal+filterCenter-i)*(BLOCKX+filterCenter*2)] = in[x+j + (y-i)*width];
                    }
                }
                else
                {
                    if(y - i >= 0)
                        data_sm[(xLocal+filterCenter) + (yLocal+filterCenter-i)*(BLOCKX+filterCenter*2)] = in[x + (y-i)*width];
                }
            }
        }
        
        // Transfer east border to shared memory
        if(threadIdx.x == BLOCKX-1)
        {
            for(int i = 1; i <= filterCenter; i++)
            {
                if(threadIdx.y == BLOCKY-1) // South East corner
                {
                    for(int j = 0; j <= filterCenter; j++)
                    {
                        if(x + i < width && y + j < height)
                            data_sm[(xLocal+filterCenter+i) + (yLocal+filterCenter+j)*(BLOCKX+filterCenter*2)] = in[x+i + (y+j)*width];
                    }
                }
                else
                {
                    if(x + i < width)
                        data_sm[(xLocal+filterCenter+i) + (yLocal+filterCenter)*(BLOCKX+filterCenter*2)] = in[x+i + y*width];
                }
            }
        }
        
        // Transfer south border to shared memory
        if(threadIdx.y == BLOCKY-1)
        {
            for(int i = 1; i <= filterCenter; i++)
            {
                if(threadIdx.x == 0) // South West corner
                {
                    for(int j = 0; j <= filterCenter; j++)
                    {
                        if(x - j >= 0 && y + i < height)
                            data_sm[(xLocal+filterCenter-j) + (yLocal+filterCenter+i)*(BLOCKX+filterCenter*2)] = in[x-j + (y+i)*width];
                    }
                }
                else
                {
                    if(y + i < height)
                        data_sm[(xLocal+filterCenter) + (yLocal+filterCenter+i)*(BLOCKX+filterCenter*2)] = in[x + (y+i)*width];
                }
            }
        }

        __syncthreads();
        int aggregate = 0;
        for (int ky = 0; ky < filterDim; ky++)
        {
            int nky = filterDim - 1 - ky;
            for (int kx = 0; kx < filterDim; kx++)
            {
                int nkx = filterDim - 1 - kx;
                int yy = yLocal + ky;
                int xx = xLocal + kx;
                int yyy = y + (ky - filterCenter);
                int xxx = x + (kx - filterCenter);
                if (xxx >= 0 && xxx < width && yyy >= 0 && yyy < height)
                    aggregate += data_sm[xx + yy*(BLOCKX+filterCenter*2)] * filter_sm[nky * filterDim + nkx];
            }
        }
        aggregate *= filterFactor;
        if (aggregate > 0)
            out[x + y*width] = (aggregate > 255) ? 255 : aggregate;
        else
            out[x + y*width] = 0;
    }
    else
        __syncthreads();
}

// GPU cooperative groups - Apply convolutional filter on image data
__global__ void device_applyFilter_cg(unsigned char *image, unsigned char *process, const int width, const int height, const int Nx, const int Ny, 
    const int filterDim, const float filterFactor, const int iterations)
{
    //__shared__ unsigned char data_sq[1]; 
    cg::grid_group grid = cg::this_grid();
    for (int i = 0; i < iterations; i++)
    {
        int x = (blockIdx.x * BLOCKX + threadIdx.x) * Nx;
        int y = (blockIdx.y * BLOCKY + threadIdx.y) * Ny;
        for(int iy = 0; iy < Ny; iy++)
        {
            for(int ix = 0; ix < Nx; ix++)
            {
                if (x < width && y < height)
                {
                    const int filterCenter = filterDim / 2;
                    int aggregate = 0;
                    for (int ky = 0; ky < filterDim; ky++)
                    {
                        int nky = filterDim - 1 - ky;
                        for (int kx = 0; kx < filterDim; kx++)
                        {
                        int nkx = filterDim - 1 - kx;
                        int yy = y + (ky - filterCenter);
                        int xx = x + (kx - filterCenter);
                        if (xx >= 0 && xx < width && yy >= 0 && yy < height)
                            aggregate += image[xx + yy*width] * constFilterGPU[nky * filterDim + nkx];
                        }
                    }
                    aggregate *= filterFactor;
                    if (aggregate > 0)
                        process[x + y*width] = (aggregate > 255) ? 255 : aggregate;
                    else
                        process[x + y*width] = 0;
                }
                x++;
            }
            y++;
        }
        cg::sync(grid);
        unsigned char *temp = process;
        *process = *image;
        *image = *temp;
        cg::sync(grid);
    }
}

void help(char const *exec, char const opt, char const *optarg)
{
    FILE *out = stdout;
    if (opt != 0)
    {
        out = stderr;
        if (optarg)
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        else
            fprintf(out, "Invalid parameter - %c\n", opt);
    }
    fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options:\n");
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");
    fprintf(out, "  -t, --test                       compare GPU and CPU code\n");
    fprintf(out, "\n");
    fprintf(out, "Example: %s in.bmp out.bmp -i 10000\n", exec);
}

double walltime(void)
{
    static struct timeval t;
    gettimeofday(&t, NULL);
    return (t.tv_sec + 1e-6 * t.tv_usec);
}

int isImageChannelEqual(unsigned char **a, unsigned char **b, const int sizeX, const int sizeY, const int ch)
{
    int errors = 0;
    for(int y = 0; y < sizeY; y++)
    {
        for(int x = 0; x < sizeX; x++)
        {
            if(a[y][x] != b[y][x])
            {
                if(errors == 0)
                    printf("\n*** Pixel errors in image ***\n");
                errors++;
                if(errors <= 5)
                printf("Ch %d: Pixel [x=%d, y=%d] \t expected %d \t got %d\n", ch, x, y, b[y][x], a[y][x]);
            }
        }
    }
    if(errors > 0)
        printf("Ch %d: %d errors!\n", ch, errors);
    return errors;
}

int getNumberOfSM(int devID)
{
    cudaDeviceProp deviceProp;
    cudaErrorCheck(cudaGetDeviceProperties(&deviceProp, devID));
    return deviceProp.multiProcessorCount;
}

void freeMemory(char *output, char *input, bmpImage *image, 
    bmpImageChannel *imageChannel1, bmpImageChannel *imageChannel2, bmpImageChannel *imageChannel3, bmpImageChannel *imageChannel4)
{
    if (output)
        free(output);
    if (input)
        free(input);
    if (image)
        freeBmpImage(image);
    if (imageChannel1)
        freeBmpImageChannel(imageChannel1);
    if (imageChannel2)
        freeBmpImageChannel(imageChannel2);
    if (imageChannel3)
        freeBmpImageChannel(imageChannel3);
    if (imageChannel4)
        freeBmpImageChannel(imageChannel4);    
}

int main(int argc, char **argv)
{
    // Walltime variables
    double startTime;
    double serialTime = 0;
    double cudaTime = 0;
    double cudaTime_sm = 0;
    double cudaTime_cg = 0;
    double activateCudaTime = 0;

    // Compare GPU and CPU code
    bool test = false;

    // Parameter parsing
    unsigned int iterations = 1;
    char *output = NULL;
    char *input = NULL;
    bmpImage *image = NULL;
    bmpImageChannel *imageChannel1 = NULL;
    bmpImageChannel *imageChannel2 = NULL;
    bmpImageChannel *imageChannel3 = NULL;
    bmpImageChannel *imageChannel4 = NULL;

    static struct option const long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"test", no_argument, 0, 't'},
        {"iterations", required_argument, 0, 'i'},
        {0, 0, 0, 0}
    };

    static char const *short_options = "hti:";
    {
        char *endptr;
        int c;
        int option_index = 0;
        while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1)
        {
            switch (c)
            {
            case 'h':
                help(argv[0], 0, NULL);
                return 0;
            case 't':
                test = true;
                break;
            case 'i':
                iterations = strtol(optarg, &endptr, 10);
                if (endptr == optarg)
                {
                    help(argv[0], c, optarg);
                    return ERROR_EXIT;
                }
                break;
            default:
                abort();
            }
        }
    }

    if (argc <= (optind + 1))
    {
        help(argv[0], ' ', "Not enough arugments");
        return ERROR_EXIT;
    }
    input = (char *)calloc(strlen(argv[optind]) + 1, sizeof(char));
    strncpy(input, argv[optind], strlen(argv[optind]));
    optind++;

    output = (char *)calloc(strlen(argv[optind]) + 1, sizeof(char));
    strncpy(output, argv[optind], strlen(argv[optind]));
    optind++;
    // End of parameter parsing!

    // Create the BMP image and load it from disk.
    image = newBmpImage(0, 0);
    if (image == NULL)
    {
        fprintf(stderr, "Could not allocate new image!\n");
        freeMemory(output, input, image, imageChannel1, imageChannel2, imageChannel3, imageChannel4);
        return ERROR_EXIT;
    }
    if (loadBmpImage(image, input) != 0)
    {
        fprintf(stderr, "Could not load bmp image '%s'!\n", input);
        freeMemory(output, input, image, imageChannel1, imageChannel2, imageChannel3, imageChannel4);
        return ERROR_EXIT;
    }

    // sizeX and sizeY for image
    const int sizeX = image->width;
    const int sizeY = image->height;

    // Offset for BLOCK/GRID size sent to the cuda kernel
    int offsetBlockX = 0;
    int offsetBlockY = 0;
    int offsetGridX = 0;
    int offsetGridY = 0;
    if (sizeX % BLOCKX)
        offsetBlockX = 1;
    if (sizeY % BLOCKY)
        offsetBlockY = 1;
    if (sizeX % (GRIDX * BLOCKX))
        offsetGridX = 1;
    if (sizeY % (GRIDY * BLOCKY))
        offsetGridY = 1;
        
    if (test)
    {
        // Create a single color channel image for CPU serial code
        imageChannel1 = newBmpImageChannel(sizeX, sizeY);
        if (imageChannel1 == NULL)
        {
            fprintf(stderr, "Could not allocate new image channel 1!\n");
            freeMemory(output, input, image, imageChannel1, imageChannel2, imageChannel3, imageChannel4);
            return ERROR_EXIT;
        }
        if (extractImageChannel(imageChannel1, image, extractAverage) != 0)
        {
            fprintf(stderr, "Could not extract image channel 1!\n");
            freeMemory(output, input, image, imageChannel1, imageChannel2, imageChannel3, imageChannel4);
            return ERROR_EXIT;
        }
    }

    // Create a single color channel image for GPU basic code
    imageChannel2 = newBmpImageChannel(sizeX, sizeY);
    if (imageChannel2 == NULL)
    {
        fprintf(stderr, "Could not allocate new image channel 2!\n");
        freeMemory(output, input, image, imageChannel1, imageChannel2, imageChannel3, imageChannel4);
        return ERROR_EXIT;
    }
    if (extractImageChannel(imageChannel2, image, extractAverage) != 0)
    {
        fprintf(stderr, "Could not extract image channel 2!\n");
        freeMemory(output, input, image, imageChannel1, imageChannel2, imageChannel3, imageChannel4);
        return ERROR_EXIT;
    }

    // Create a single color channel image for GPU shared memory code
    imageChannel3 = newBmpImageChannel(sizeX, sizeY);
    if (imageChannel3 == NULL)
    {
        fprintf(stderr, "Could not allocate new image channel 2!\n");
        freeMemory(output, input, image, imageChannel1, imageChannel2, imageChannel3, imageChannel4);
        return ERROR_EXIT;
    }
    if (extractImageChannel(imageChannel3, image, extractAverage) != 0)
    {
        fprintf(stderr, "Could not extract image channel 2!\n");
        freeMemory(output, input, image, imageChannel1, imageChannel2, imageChannel3, imageChannel4);
        return ERROR_EXIT;
    }

    // Create a single color channel image for GPU cooperative groups code
    imageChannel4 = newBmpImageChannel(sizeX, sizeY);
    if (imageChannel3 == NULL)
    {
        fprintf(stderr, "Could not allocate new image channel 2!\n");
        freeMemory(output, input, image, imageChannel1, imageChannel2, imageChannel3, imageChannel4);
        return ERROR_EXIT;
    }
    if (extractImageChannel(imageChannel4, image, extractAverage) != 0)
    {
        fprintf(stderr, "Could not extract image channel 2!\n");
        freeMemory(output, input, image, imageChannel1, imageChannel2, imageChannel3, imageChannel4);
        return ERROR_EXIT;
    }

    // Activate CUDA - No delay inside work later
    startTime = walltime();
    unsigned char *dummy;
    cudaErrorCheck(cudaMalloc((void**)&dummy, sizeof(unsigned char)));
    cudaErrorCheck(cudaFree(dummy));
    activateCudaTime = walltime() - startTime;

    // Choose filter
    const int *filter = laplacian1Filter;
    const int filterDim = 3;
    const float filterFactor = laplacian1FilterFactor;

    if (test)
    {
        //********************************* CPU serial work start *********************************
        startTime = walltime();

        // CPU computation!
        bmpImageChannel *processImageChannel = newBmpImageChannel(sizeX, sizeY);
        for (unsigned int i = 0; i < iterations; i++)
        {
            applyFilter(
                processImageChannel->data,
                imageChannel1->data,
                sizeX, sizeY,
                filter, filterDim, filterFactor
            );
            unsigned char **tmp = processImageChannel->data;
            processImageChannel->data = imageChannel1->data;
            imageChannel1->data = tmp;
            unsigned char *tmp_raw = processImageChannel->rawdata;
            processImageChannel->rawdata = imageChannel1->rawdata;
            imageChannel1->rawdata = tmp_raw;
        }
        freeBmpImageChannel(processImageChannel);

        serialTime = walltime() - startTime;
        //********************************* CPU serial work stop *********************************
    }

    //********************************* GPU basic work start *********************************
    startTime = walltime();

    // Variables
    dim3 gridBlock(sizeX/BLOCKX + offsetBlockX, sizeY/BLOCKY + offsetBlockY);
    dim3 threadBlock(BLOCKX, BLOCKY);
    unsigned char *imageChannelGPU = NULL;
    unsigned char *processImageChannelGPU = NULL;
    int *filterGPU = NULL;

    // Set up device memory
    cudaErrorCheck(cudaMalloc((void**)&imageChannelGPU, sizeX*sizeY * sizeof(unsigned char)));
    cudaErrorCheck(cudaMalloc((void**)&processImageChannelGPU, sizeX*sizeY * sizeof(unsigned char)));
    cudaErrorCheck(cudaMalloc((void**)&filterGPU, filterDim*filterDim * sizeof(int)));

    // Copy data from host to device
    cudaErrorCheck(cudaMemcpy(imageChannelGPU, imageChannel2->rawdata, sizeX*sizeY * sizeof(unsigned char), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(filterGPU, filter, filterDim*filterDim * sizeof(int), cudaMemcpyHostToDevice));

    // GPU computation
    for (unsigned int i = 0; i < iterations; i++)
    {
        device_applyFilter<<<gridBlock, threadBlock>>>(
            processImageChannelGPU, 
            imageChannelGPU,
            sizeX, sizeY, 
            filterGPU, filterDim, filterFactor
        );
        cudaErrorCheck(cudaGetLastError());
        unsigned char *temp = processImageChannelGPU;
        processImageChannelGPU = imageChannelGPU;
        imageChannelGPU = temp;
    }

    // Copy data from device to host
    cudaErrorCheck(cudaMemcpy(imageChannel2->rawdata, imageChannelGPU, sizeX*sizeY * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Free the device memory
    cudaErrorCheck(cudaFree(imageChannelGPU));
    cudaErrorCheck(cudaFree(processImageChannelGPU));
    cudaErrorCheck(cudaFree(filterGPU));

    cudaTime = walltime() - startTime;
    //********************************* GPU basic work stop *********************************

    //********************************* GPU shared memory work start *********************************
    startTime = walltime();

    // Variables
    dim3 gridBlock_sm(sizeX/BLOCKX + offsetBlockX, sizeY/BLOCKY + offsetBlockY);
    dim3 threadBlock_sm(BLOCKX, BLOCKY);
    unsigned char *imageChannelGPU_sm = NULL;
    unsigned char *processImageChannelGPU_sm = NULL;
    int *filterGPU_sm = NULL;
    const int N = (BLOCKX+2*(filterDim/2))*(BLOCKY+2*(filterDim/2));
    const int sizeImage_sm = N * sizeof(unsigned char);
    const int sizeFilter_sm = filterDim * filterDim * sizeof(int);
    const int size_sm = sizeImage_sm + sizeFilter_sm;

    // Set up device memory
    cudaErrorCheck(cudaMalloc((void**)&imageChannelGPU_sm, sizeX*sizeY * sizeof(unsigned char)));
    cudaErrorCheck(cudaMalloc((void**)&processImageChannelGPU_sm, sizeX*sizeY * sizeof(unsigned char)));
    cudaErrorCheck(cudaMalloc((void**)&filterGPU_sm, filterDim*filterDim * sizeof(int)));

    // Copy data from host to device
    cudaErrorCheck(cudaMemcpy(imageChannelGPU_sm, imageChannel3->rawdata, sizeX*sizeY * sizeof(unsigned char), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(filterGPU_sm, filter, filterDim*filterDim * sizeof(int), cudaMemcpyHostToDevice));

    // GPU computation
    for (unsigned int i = 0; i < iterations; i++)
    {
        device_applyFilter_sm<<<gridBlock_sm, threadBlock_sm, size_sm>>>(
            processImageChannelGPU_sm, 
            imageChannelGPU_sm,
            sizeX, sizeY, N, 
            filterGPU_sm, filterDim, filterFactor
        );
        cudaErrorCheck(cudaGetLastError());
        unsigned char *temp_sm = processImageChannelGPU_sm;
        processImageChannelGPU_sm = imageChannelGPU_sm;
        imageChannelGPU_sm = temp_sm;
    }

    // Copy data from device to host
    cudaErrorCheck(cudaMemcpy(imageChannel3->rawdata, imageChannelGPU_sm, sizeX*sizeY * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Free the device memory
    cudaErrorCheck(cudaFree(imageChannelGPU_sm));
    cudaErrorCheck(cudaFree(processImageChannelGPU_sm));
    cudaErrorCheck(cudaFree(filterGPU_sm));
    
    cudaTime_sm = walltime() - startTime;
    //********************************* GPU shared memory work stop *********************************

    //********************************* GPU cooperative groups work start ***************************
    startTime = walltime();

    // Variables
    dim3 gridDim_cg(GRIDX, GRIDY);
    dim3 blockDim_cg(BLOCKX, BLOCKY);
    unsigned char *imageChannelGPU_cg = NULL;
    unsigned char *processImageChannelGPU_cg = NULL;
    const int sizeFilter_cg = filterDim * filterDim * sizeof(int);
    const int Nx = sizeX / (GRIDX * BLOCKX) + offsetGridX;
    const int Ny = sizeY / (GRIDY * BLOCKY) + offsetGridY;

    // Set up device memory
    cudaErrorCheck(cudaMalloc((void**)&imageChannelGPU_cg, sizeX*sizeY * sizeof(unsigned char)));
    cudaErrorCheck(cudaMalloc((void**)&processImageChannelGPU_cg, sizeX*sizeY * sizeof(unsigned char)));

    // Copy data from host to device
    cudaErrorCheck(cudaMemcpy(imageChannelGPU_cg, imageChannel3->rawdata, sizeX*sizeY * sizeof(unsigned char), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpyToSymbol(constFilterGPU, filter, sizeFilter_cg));

    // Arguments for CUDA kernel
    void *kernelArgs_cg[] = {
        (void *)&imageChannelGPU_cg, (void *)&processImageChannelGPU_cg,
        (void *)&sizeX, (void *)&sizeY,
        (void *)&Nx, (void *)&Ny,
        (void *)&filterDim, (void *)&filterFactor,
        (void *)&iterations
    };

    // GPU computation
    cudaErrorCheck(cudaLaunchCooperativeKernel(
        (void *)device_applyFilter_cg,
        gridDim_cg, blockDim_cg, 
        kernelArgs_cg,                                      
        0, 
        NULL
    ));
    cudaErrorCheck(cudaGetLastError());

    // Copy data from device to host
    cudaErrorCheck(cudaMemcpy(imageChannel4->rawdata, imageChannelGPU_cg, sizeX*sizeY * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Free the device memory
    cudaErrorCheck(cudaFree(imageChannelGPU_cg));
    cudaErrorCheck(cudaFree(processImageChannelGPU_cg));
    
    cudaTime_cg = walltime() - startTime;
    //********************************* GPU cooperative groups work stop ****************************

    if (test)
    {
        // Check if GPU image channel is equal to CPU image channel
        int errors = 0;
        errors += isImageChannelEqual(imageChannel2->data, imageChannel1->data, sizeX, sizeY, 2);
        errors += isImageChannelEqual(imageChannel3->data, imageChannel1->data, sizeX, sizeY, 3);
        errors += isImageChannelEqual(imageChannel4->data, imageChannel1->data, sizeX, sizeY, 4);
        if(errors == 0)
            printf("\n*** Every pixel is correct! ***\n");
    }

    // Map our single color image back to a normal BMP image with 3 color channels
    if (mapImageChannel(image, imageChannel4, mapEqual) != 0)
    {
        fprintf(stderr, "Could not map image channel!\n");
        freeMemory(output, input, image, imageChannel1, imageChannel2, imageChannel3, imageChannel4);
        return ERROR_EXIT;
    }

    //Write the image back to disk
    if (saveBmpImage(image, output) != 0)
    {
        fprintf(stderr, "Could not save output to '%s'!\n", output);
        freeMemory(output, input, image, imageChannel1, imageChannel2, imageChannel3, imageChannel4);
        return ERROR_EXIT;
    };

    printf("\n*** Run times ***\n");
    printf("Running with %d iteration(s)\n", iterations);
    printf("Activate CUDA time:%12.3f ms\n", activateCudaTime * 1e3);
    printf("     Work GPU time:%12.3f ms\tCooperative groups \tch 4 \n", cudaTime_cg * 1e3);
    printf("     Work GPU time:%12.3f ms\tShared memory \t\tch 3 \n", cudaTime_sm * 1e3);
    printf("     Work GPU time:%12.3f ms\tBasic \t\t\tch 2 \n", cudaTime * 1e3);

    if(test)
    {
        printf("     Work CPU time:%12.3f ms\tSerial \t\t\tch 1 \n", serialTime * 1e3);
        printf("\n*** Speedups ***\n");
        printf("Cooperative groups GPU: %6.1f times faster then shared memory GPU\n", cudaTime_sm/cudaTime_cg);
        printf("Cooperative groups GPU: %6.1f times faster then basic GPU\n", cudaTime/cudaTime_cg);
        printf("Cooperative groups GPU: %6.1f times faster then serial CPU\n", serialTime/cudaTime_cg);
        printf("     Shared memory GPU: %6.1f times faster then basic GPU\n", cudaTime/cudaTime_sm);
        printf("     Shared memory GPU: %6.1f times faster then serial CPU\n", serialTime/cudaTime_sm);
        printf("             Basic GPU: %6.1f times faster then serial CPU\n\n", serialTime/cudaTime);
    }
    else
        printf("\n");

    freeMemory(output, input, image, imageChannel1, imageChannel2, imageChannel3, imageChannel4);
    return 0;
};
