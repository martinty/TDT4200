
#include <getopt.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#include <complex.h>
#include <time.h>
#include "libs/bitmap.h"
#include "libs/utilities.h"
#include "main.h"

// colourGradientSteps is defined in main.h
// first step MUST be 0.0
// last step MUST be 1.0
static const struct colourGradientStep colourGradient[colourGradientSteps] = {
	{.step = 0.0, {.r = 0, .g = 0, .b = 0}},
	{.step = 0.03, {.r = 0, .g = 7, .b = 100}},
	{.step = 0.16, {.r = 32, .g = 107, .b = 203}},
	{.step = 0.42, {.r = 237, .g = 255, .b = 255}},
	{.step = 0.64, {.r = 255, .g = 170, .b = 0}},
	{.step = 0.86, {.r = 0, .g = 2, .b = 0}},
	{.step = 1.0, {.r = 0, .g = 0, .b = 0}}};

// Gets initialized after parameters are parsed
static pixel *colourMap = NULL;
static unsigned int colourMapSize = 0;

static unsigned int res = 2048;
static unsigned int maxDwell = 512;

struct timespec timeStart, timeEnd;

pixel getDwellColour(unsigned int const y, unsigned int const x, unsigned long long const dwell)
{
	static const double log2 = 0.693147180559945309417232121458176568075500134360255254120;
	double complex z = x + y * I;
	unsigned long long index = dwell + 1 - log(log(cabs(z) / log2));
	return colourMap[index % colourMapSize];
}

double complex getInitialValue(double complex const cmin, double complex const cmax, unsigned int const y, unsigned int const x)
{
	double real = ((double)x / res) * creal(cmax - cmin) + creal(cmin);
	double imag = ((double)y / res) * cimag(cmax - cmin) + cimag(cmin);
	double complex ret = real + imag * I;
	return ret;
}

unsigned long long pixelDwell(double complex const cmin, double complex const cmax, unsigned int const y, unsigned int const x)
{
	double complex const init = getInitialValue(cmin, cmax, y, x);
	double complex z = init;
	unsigned long long dwell = 0;

	while (dwell < maxDwell && cabs(z) < 4.0)
	{
		z = z * z + init;
		dwell++;
	}

	return dwell;
}

void computeAndMapDwell(bmpImage *image, double complex cmin, double complex cmax)
{
	for (unsigned int y = 0; y < res; y++)
	{
		for (unsigned int x = 0; x < res; x++)
		{
			image->data[y][x] = getDwellColour(y, x, pixelDwell(cmin, cmax, y, x));
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
	fprintf(out, "Mandelbrot Set Renderer\n\n");
	fprintf(out, "  -x [0;1]         Center of Re[-1.5;0.5] (default=0.5)\n");
	fprintf(out, "  -y [0;1]         Center of Im[-1;1] (default=0.5)\n");
	fprintf(out, "  -s (0;1]         Inverse scaling factor (default=1)\n");
	fprintf(out, "  -r [pixel]       Image resolution (default=2048)\n");
	fprintf(out, "  -i [iterations]  Iterations or max dwell (default=512)\n");
	fprintf(out, "  -c [colours]     Colour map iterations (default=1)\n");
	fprintf(out, "%s [options]  <output-bmp>\n", exec);
}

int main(int argc, char *argv[])
{
	int ret = 1;
	/* Standard Values */
	char *output = NULL;			   //output image filepath
	bmpImage *image = NULL;			   //output image
	double x = 0.5, y = 0.5;		   // coordinates
	double scale = 1;				   // scaling factor
	unsigned int colourIterations = 1; //how many times the colour gradient is repeated
	bool quiet = false;				   //output something or not

	/* Parameter parsing... */
	{
		char c;
		while ((c = getopt(argc, argv, "x:y:s:r:i:c:qh")) != -1)
		{
			switch (c)
			{
			case 'x':
				x = clampDouble(atof(optarg), 0.0, 1.0);
				break;
			case 'y':
				y = clampDouble(atof(optarg), 0.0, 1.0);
				break;
			case 's':
				scale = clampDouble(atof(optarg), 0.0, 1.0);
				if (scale == 0)
					scale = 1;
				break;
			case 'r':
				res = atoi(optarg);
				break;
			case 'i':
				maxDwell = atoi(optarg);
				break;
			case 'c':
				colourIterations = atoi(optarg);
				break;
			case 'q':
				quiet = true;
				break;
			case 'h':
				help(argv[0], 0, NULL);
				goto exit_graceful;
				break;
			default:
				abort();
			}
		}
	}

	if (argc <= (optind))
	{
		help(argv[0], ' ', "Not enough arugments");
		goto error_exit;
	}

	output = calloc(strlen(argv[optind]) + 1, sizeof(char));
	strncpy(output, argv[optind], strlen(argv[optind]));
	optind++;

	/* Initialize the colourMap, which assigns each possible dwell Value a colour */
	/* This might look complex but is just eye candy. */
	/* Could be made much simpler, thus less fancy */

	colourMapSize = maxDwell / colourIterations;
	colourMap = malloc(colourMapSize * sizeof(pixel));
	initColourMap(colourMap, colourMapSize, colourGradient);

	/* Putting some magic numbers in place...  */
	double const xmin = -3.5 + (2 * 2 * x);
	double const xmax = -1.5 + (2 * 2 * x);
	double const ymin = -3.0 + (2 * 2 * y);
	double const ymax = -1.0 + (2 * 2 * y);
	double const xlen = fabs(xmin - xmax);
	double const ylen = fabs(ymin - ymax);

	/* This is our complex area that is going to be plotted */
	double complex const cmin = (xmin + (0.5 * (1 - scale) * xlen)) + (ymin + (0.5 * (1 - scale) * ylen)) * I;
	double complex const cmax = (xmax - (0.5 * (1 - scale) * xlen)) + (ymax - (0.5 * (1 - scale) * ylen)) * I;

	/* Output useful informations... */
	if (!quiet)
	{
		printf("Center:      [%f,%f]\n", x, y);
		printf("Zoom:        %llu%%\n", (unsigned long long)(1 / scale) * 100);
		printf("Iterations:  %u\n", maxDwell);
		printf("Window:      Re[%f,%f], Im[%f,%f]\n", creal(cmin), creal(cmax), cimag(cmin), cimag(cmax));
		printf("Output:      %s\n", output);
	}

	//Allocate bmp image
	image = newBmpImage(res, res);
	if (image == NULL)
	{
		fprintf(stderr, "ERROR: could not allocate bmp image space!\n");
		goto error_exit;
	}

	// Compute the dwell and map it to the bmpImage with fancy colors
	clock_gettime(CLOCK_REALTIME, &timeStart);
	computeAndMapDwell(image, cmin, cmax);
	clock_gettime(CLOCK_REALTIME, &timeEnd);

	// Save the Image
	if (saveBmpImage(image, output))
	{
		fprintf(stderr, "ERROR: could not save image to %s\n", output);
		goto error_exit;
	}

	// Print execution time for computeAndMapDwell()
	double timeElapsed = (double)(timeEnd.tv_sec - timeStart.tv_sec) + 
						 (double)(timeEnd.tv_nsec - timeStart.tv_nsec) / 1000000000;
	printf("Execution time for computeAndMapDwell(): %0.3f seconds \n", timeElapsed); 

	goto exit_graceful;
error_exit:
	ret = 1;
exit_graceful:
	if (image)
		freeBmpImage(image);
	if (colourMap)
		free(colourMap);
	return 0;
}
