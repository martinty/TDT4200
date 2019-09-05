#ifndef BITMAP_H
#define BITMAP_H


typedef unsigned char uchar;
typedef struct {
    uchar r;
    uchar g;
    uchar b;
} Pixel;

void readbmp(char *filename, Pixel *array);
void savebmp(char *name, Pixel *buffer, int x, int y);

void invertColor(Pixel *array, int x, int y);
void flipLeftRight(Pixel *array, int x, int y);
void doubleImageSize(Pixel *array, Pixel *newArray, int x, int y);

#endif
