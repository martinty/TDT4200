#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

int main() {
	uchar *image = calloc(XSIZE * YSIZE * 3, 1); // Three uchars per pixel (RGB)
	readbmp("before.bmp", image);


	// Alter the image here
	flipLeftRight(image, XSIZE, YSIZE);
	flipUpDown(image, XSIZE, YSIZE);
	invertColor(image, XSIZE, YSIZE);
	saveDoubleSizeBmp("afterDouble.bmp", image, XSIZE, YSIZE);


	savebmp("after.bmp", image, XSIZE, YSIZE);
	free(image);
	return 0;
}