#include <stdlib.h>
#include <stdio.h>
#include "bitmap.h"

// save 24-bits bmp file, buffer must be in bmp format: upside-down
void savebmp(char *name,uchar *buffer,int x,int y) {
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
	fwrite(buffer,1,x*y*3,f);
	fclose(f);
	printf("Image saved as %s \n", name);
}

// read bmp file and store image in contiguous array
void readbmp(char* filename, uchar* array) {
	FILE* img = fopen(filename, "rb");   //read the file
	if(!img) {
		printf("Error: Can't open image \n");
		exit(1);
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

	for (int i=0; i<height; i++ ) {
		fread( data, sizeof(uchar), widthnew, img);
		for (int j=0; j<width*3; j+=3) {
			array[3 * i * width + j + 0] = data[j+0];
			array[3 * i * width + j + 1] = data[j+1];
			array[3 * i * width + j + 2] = data[j+2];
		}
	}
	fclose(img); //close the file
}

void flipLeftRight(uchar *array, int width, int height){
	uchar lefSideTemp;
	for(int y = 0; y < height; y++){
		for(int x = 0; x < width*3/2; x += 3){
			for(int i = 0; i < 3; i++){
				lefSideTemp = array[3 * y * width + x + i]; 
				array[3 * y * width + x + i] = array[3 * y * width + (3 * width - 3 - x) + i];
				array[3 * y * width + (3 * width - 3 - x) + i] = lefSideTemp;
			}
		}
	}
}

void flipUpDown(uchar *array, int width, int height){
	uchar upSideTemp;
	for(int y = 0; y < height/2; y++){
		for(int x = 0; x < width*3; x += 3){
			for(int i = 0; i < 3; i++){
				upSideTemp = array[3 * y * width + x + i]; 
				array[3 * y * width + x + i] = array[3 * (height - 1 - y) * width + x + i];
				array[3 * (height - 1 - y) * width + x + i] = upSideTemp;
			}
		}
	}
}

void invertColor(uchar *array, int width, int height){
	for(int y = 0; y < height; y++){
		for(int x = 0; x < width*3; x += 3){
			for(int i = 0; i < 3; i++){
				array[3 * y * width + x + i] = 255 - array[3 * y * width + x + i]; 
			}
		}
	}
}

void saveDoubleSizeBmp(char *name, uchar *array, int width, int height){
	int newWidth = width * 2;
	int newHeight = height * 2;
	uchar *newImage = calloc(newWidth * newHeight * 3, 1);
	for(int y = 0; y < height; y++){
		for(int x = 0; x < width*3; x += 3){
			for(int i = 0; i < 3; i++){
				uchar color = array[3 * y * width + x + i];
				newImage[3 * (y*2) * newWidth + (x*2) + i] = color;
				newImage[3 * (y*2) * newWidth + (x*2+3) + i] = color;
				newImage[3 * (y*2+1) * newWidth + (x*2) + i] = color;
				newImage[3 * (y*2+1) * newWidth + (x*2+3) + i] = color;
			}
		}
	}
	savebmp(name, newImage, newWidth, newHeight);
	free(newImage);
}