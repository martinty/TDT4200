#ifndef BORDERS_H
#define BORDERS_H

void exchangeHorizontalBorders(bmpImageChannel *imageChannel, int rank, int size);
void exchangeSouthBorder(bmpImageChannel *imageChannel, int rank, int size);
void exchangeNorthBorder(bmpImageChannel *imageChannel, int rank);

#endif
