#ifndef BITMAP_H
#define BITMAP_H


typedef unsigned char uchar;
void savebmp(char *name, uchar *buffer, int x, int y);
void readbmp(char *filename, uchar *array);

void flipLeftRight(uchar *array, int width, int height);
void flipUpDown(uchar *array, int width, int height);
void invertColor(uchar *array, int width, int height);
void saveDoubleSizeBmp(char *name, uchar *array, int width, int height);

#endif
