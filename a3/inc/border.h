#ifndef BORDER_H
#define BORDER_H

typedef struct
{
    unsigned char *north;
    unsigned char *east;
    unsigned char *south;
    unsigned char *west;
} borders;

borders *newBorders(unsigned int const width, unsigned int const height, int n, int e, int s, int w);
void freeBorders(borders *ghostCells);

void exchangeHorizontalBorders(bmpImageChannel *imageChannel, borders *ghostCells, int rank, int size);
void exchangeSouthBorder(bmpImageChannel *imageChannel, borders *ghostCells, int rank, int size);
void exchangeNorthBorder(bmpImageChannel *imageChannel, borders *ghostCells, int rank);

#endif
