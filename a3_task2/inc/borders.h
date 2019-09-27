#ifndef BORDERS_H
#define BORDERS_H

void exchangeHorizontalBorders(bmpImageChannel *imageChannel, int ghostRows, int rank, int size);
void exchangeSouthBorder(bmpImageChannel *imageChannel, int ghostRows, int rank, int size);
void exchangeNorthBorder(bmpImageChannel *imageChannel, int ghostRows, int rank);

#endif
