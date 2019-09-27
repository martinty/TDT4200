#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <mpi.h>
#include "bitmap.h"
#include "borders.h"

// Convolutional Kernel Examples, each with dimension 3,
// gaussian kernel with dimension 5
// If you apply another kernel, remember not only to exchange
// the kernel but also the kernelFactor and the correct dimension.

int const sobelYKernel[] = {-1, -2, -1,
                            0, 0, 0,
                            1, 2, 1};
float const sobelYKernelFactor = (float)1.0;

int const sobelXKernel[] = {-1, 0, 1,
                            -2, 0, 2,
                            -1, 0, 1};
float const sobelXKernelFactor = (float)1.0;

int const laplacian1Kernel[] = {-1, -4, -1,
                                -4, 20, -4,
                                -1, -4, -1};

float const laplacian1KernelFactor = (float)1.0;

int const laplacian2Kernel[] = {0, 1, 0,
                                1, -4, 1,
                                0, 1, 0};
float const laplacian2KernelFactor = (float)1.0;

int const laplacian3Kernel[] = {-1, -1, -1,
                                -1, 8, -1,
                                -1, -1, -1};
float const laplacian3KernelFactor = (float)1.0;

// Bonus Kernel:

int const gaussianKernel[] = {1, 4, 6, 4, 1,
                              4, 16, 24, 16, 4,
                              6, 24, 36, 24, 6,
                              4, 16, 24, 16, 4,
                              1, 4, 6, 4, 1};

float const gaussianKernelFactor = (float)1.0 / 256.0;

// Helper function to swap bmpImageChannel pointers
void swapImageChannel(bmpImageChannel **one, bmpImageChannel **two)
{
    bmpImageChannel *helper = *two;
    *two = *one;
    *one = helper;
}

// Apply convolutional kernel on image data
void applyKernel(unsigned char **out, unsigned char **in, unsigned int width, unsigned int height, int *kernel, unsigned int kernelDim, float kernelFactor)
{
    unsigned int const kernelCenter = (kernelDim / 2);
    for (unsigned int y = 0; y < height; y++)
    {
        for (unsigned int x = 0; x < width; x++)
        {
            int aggregate = 0;
            for (unsigned int ky = 0; ky < kernelDim; ky++)
            {
                int nky = kernelDim - 1 - ky;
                for (unsigned int kx = 0; kx < kernelDim; kx++)
                {
                    int nkx = kernelDim - 1 - kx;

                    int yy = y + (ky - kernelCenter);
                    int xx = x + (kx - kernelCenter);
                    if (xx >= 0 && xx < (int)width && yy >= 0 && yy < (int)height)
                        aggregate += in[yy][xx] * kernel[nky * kernelDim + nkx];
                }
            }
            aggregate *= kernelFactor;
            if (aggregate > 0)
            {
                out[y][x] = (aggregate > 255) ? 255 : aggregate;
            }
            else
            {
                out[y][x] = 0;
            }
        }
    }
}

void help(char const *exec, char const opt, char const *optarg)
{
    FILE *out = stdout;
    if (opt != 0)
    {
        out = stderr;
        if (optarg)
        {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        }
        else
        {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }
    fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options:\n");
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s in.bmp out.bmp -i 10000\n", exec);
}

void errorExit(char *output, char *input, bmpImage *image, bmpImage *localImage, bmpImageChannel *imageChannel, int status)
{
    if (imageChannel)
        freeBmpImageChannel(imageChannel);
    if (localImage)
        freeBmpImage(localImage);
    if (image)
        free(image);
    if (input)
        free(input);
    if (output)
        free(output);
    exit(status);
}

int main(int argc, char **argv)
{
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Make Pixel datatype
    MPI_Datatype pixel_dt;
    MPI_Type_contiguous(3, MPI_UNSIGNED_CHAR, &pixel_dt);
    MPI_Type_commit(&pixel_dt);

    // Make info datatype
    MPI_Datatype info_dt;
    MPI_Type_contiguous(3, MPI_UNSIGNED, &info_dt);
    MPI_Type_commit(&info_dt);

    // Image
    bmpImage *image = newBmpImage(0, 0);
    char *output = NULL;
    char *input = NULL;
    bmpImage *localImage = NULL;
    bmpImageChannel *imageChannel = NULL;
    information info;


    if (world_rank == 0)
    {
        // Parameter parsing, don't change this!
        unsigned int iterations = 1;

        static struct option const long_options[] = {
            {"help", no_argument, 0, 'h'},
            {"iterations", required_argument, 0, 'i'},
            {0, 0, 0, 0}};

        static char const *short_options = "hi:";
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
                    errorExit(output, input, image, localImage, imageChannel, 0);
                case 'i':
                    iterations = strtol(optarg, &endptr, 10);
                    if (endptr == optarg)
                    {
                        help(argv[0], c, optarg);
                        errorExit(output, input, image, localImage, imageChannel, 1);
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
            errorExit(output, input, image, localImage, imageChannel, 1);
        }
        input = calloc(strlen(argv[optind]) + 1, sizeof(char));
        strncpy(input, argv[optind], strlen(argv[optind]));
        optind++;

        output = calloc(strlen(argv[optind]) + 1, sizeof(char));
        strncpy(output, argv[optind], strlen(argv[optind]));
        optind++;
        // End of Parameter parsing!

        // Load BMP image from disk.
        if (loadBmpImage(image, input) != 0)
        {
            fprintf(stderr, "Could not load bmp image '%s'!\n", input);
            errorExit(output, input, image, localImage, imageChannel, 1);
        }

        // Update info
        info.iterations = iterations;
        info.imageWidth = image->width;
        info.imageHeight = image->height;
    }

    MPI_Bcast(&info, 1, info_dt, 0, MPI_COMM_WORLD);

    int sendCounts[world_size];
    int displs[world_size];
    int heightScale[world_size];
    int offset = 0;
    int kernelDim = 3; // Need to be the same as kernelDim in applyKernel()
    int ghostRows = kernelDim/2;

    heightScale[0] = info.imageHeight / world_size + info.imageHeight % world_size;
    sendCounts[0] = heightScale[0] * info.imageWidth;
    displs[0] = offset;
    offset += sendCounts[0];

    for (int i = 1; i < world_size; i++)
    {
        heightScale[i] = info.imageHeight / world_size;
        sendCounts[i] = heightScale[i] * info.imageWidth;
        displs[i] = offset;
        offset += sendCounts[i];
    }

    localImage = newBmpImage(info.imageWidth, heightScale[world_rank] + ghostRows*2);
    MPI_Scatterv(image->rawdata, sendCounts, displs, pixel_dt, localImage->rawdata + info.imageWidth*ghostRows, sendCounts[world_rank], pixel_dt, 0, MPI_COMM_WORLD);
 Signal code: Invalid permissions (2)
    //  *** Work start ***

    // Create a single color channel image. It is easier to work just with one color
    imageChannel = newBmpImageChannel(localImage->width, localImage->height);
    if (imageChannel == NULL)
    {
        fprintf(stderr, "Could not allocate new image channel!\n");
        errorExit(output, input, image, localImage, imageChannel, 1);
    }

    // Extract from the loaded image an average over all colors - nothing else than
    // a black and white representation
    // extractImageChannel and mapImageChannel need the images to be in the exact
    // same dimensions!
    // Other prepared extraction functions are extractRed, extractGreen, extractBlue
    if (extractImageChannel(imageChannel, localImage, extractAverage) != 0)
    {
        fprintf(stderr, "Could not extract image channel!\n");
        errorExit(output, input, image, localImage, imageChannel, 1);
    }

    // Here we do the actual computation!
    // imageChannel->data is a 2-dimensional array of unsigned char which is accessed row first ([y][x])
    bmpImageChannel *processImageChannel = newBmpImageChannel(imageChannel->width, imageChannel->height);
    for (unsigned int i = 0; i < info.iterations; i++)
    {
        // Exchange borders
        exchangeHorizontalBorders(imageChannel, ghostRows, world_rank, world_size);

        applyKernel(processImageChannel->data,
                    imageChannel->data,
                    imageChannel->width,
                    imageChannel->height,
                    (int *)laplacian1Kernel, 3, laplacian1KernelFactor
                    //(int *)laplacian2Kernel, 3, laplacian2KernelFactor
                    //(int *)laplacian3Kernel, 3, laplacian3KernelFactor
                    //(int *)gaussianKernel, 5, gaussianKernelFactor
        );
        swapImageChannel(&processImageChannel, &imageChannel);
    }
    freeBmpImageChannel(processImageChannel);

    // Map our single color image back to a normal BMP image with 3 color channels
    // mapEqual puts the color value on all three channels the same way
    // other mapping functions are mapRed, mapGreen, mapBlue
    if (mapImageChannel(localImage, imageChannel, mapEqual) != 0)
    {
        fprintf(stderr, "Could not map image channel!\n");
        errorExit(output, input, image, localImage, imageChannel, 1);
    }

    // *** Work stop ***

    MPI_Gatherv(localImage->rawdata + info.imageWidth*ghostRows, sendCounts[world_rank], pixel_dt, image->rawdata, sendCounts, displs, pixel_dt, 0, MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        //Write the image back to disk
        if (saveBmpImage(image, output) != 0)
        {
            fprintf(stderr, "Could not save output to '%s'!\n", output);
            errorExit(output, input, image, localImage, imageChannel, 1);
        }
    }

    // Free data
    if (imageChannel)
        freeBmpImageChannel(imageChannel);
    if (localImage)
        freeBmpImage(localImage);
    if (image)
        freeBmpImage(image);
    if (input)
        free(input);
    if (output)
        free(output);

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
};