#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"

// save 24-bits bmp file, buffer must be in bmp format: upside-down
void savebmp(char *name, Pixel *buffer, int x, int y) {
	FILE *f=fopen(name,"wb");
	if(!f) {
		printf("Error writing image to disk.\n");
		return;
	}
	unsigned int size=x*y*3+54;
	uchar header[54]={'B','M',size&255,(size>>8)&255,(size>>16)&255,size>>24,0,
                    0,0,0,54,0,0,0,40,0,0,0,x&255,x>>8,0,0,y&255,y>>8,0,0,1,0,24,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	fwrite(header,1,54,f);
	fwrite(buffer,sizeof(Pixel),x*y,f);
	fclose(f);
}

// read bmp file and store image in contiguous array
void readbmp(char* filename, Pixel* array) {
	FILE* img = fopen(filename, "rb");   //read the file
	if(!img){
		printf("Error reading image from disk.\n");
		return;
	}
	uchar header[54];
	fread(header, sizeof(uchar), 54, img); // read the 54-byte header

  // extract image height and width from header
	int width = *(int*)&header[18];
	int height = *(int*)&header[22];
	int padding=0;
	while ((width*3+padding) % 4!=0) padding++;

	int widthnew=width*3+padding;
	uchar* data = calloc(widthnew, sizeof(uchar));

	for (int row=0; row<height; row++ ) {
		fread(data, sizeof(uchar), widthnew, img);
		for (int col=0; col<width; col++) {
			array[row*width + col].r = data[col*3+0];
			array[row*width + col].g = data[col*3+1];
			array[row*width + col].b = data[col*3+2];
		}
	}
	fclose(img); //close the file
}

void invertColor(Pixel *array, int x, int y){
	for(int row = 0; row < y; row++){
		for(int col = 0; col < x; col++){
			array[row*x + col].r = 255 - array[row*x + col].r;
			array[row*x + col].g = 255 - array[row*x + col].g;
			array[row*x + col].b = 255 - array[row*x + col].b;
		}
	}
}