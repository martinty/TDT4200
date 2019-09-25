#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "bitmap.h"
#include "border.h"

borders *newBorders(unsigned int const width, unsigned int const height, int n, int e, int s, int w)
{
    borders *new = malloc(sizeof(borders));
    new->north = calloc(width * height * n, sizeof(unsigned char));
    new->east = calloc(width * height * e, sizeof(unsigned char));
    new->south = calloc(width * height * s, sizeof(unsigned char));
    new->west = calloc(width * height * w, sizeof(unsigned char));
    return new;
}

void freeBorders(borders *ghostCells)
{
    free(ghostCells->north);
    free(ghostCells->east);
    free(ghostCells->south);
    free(ghostCells->west);
    free(ghostCells);
}

void exchangeHorizontalBorders(bmpImageChannel *imageChannel, borders *ghostCells, int rank, int size)
{
    if (rank % 2 == 0)
    {
        exchangeSouthBorder(imageChannel, ghostCells, rank, size);
        exchangeNorthBorder(imageChannel, ghostCells, rank);
    }
    else
    {
        exchangeNorthBorder(imageChannel, ghostCells, rank);
        exchangeSouthBorder(imageChannel, ghostCells, rank, size);
    }
}

void exchangeSouthBorder(bmpImageChannel *imageChannel, borders *ghostCells, int rank, int size)
{
    if (rank != size - 1)
    {
        // Send south border to rank+1 and recv north border from rank+1 as new south border
        MPI_Sendrecv((imageChannel->rawdata + imageChannel->width * (imageChannel->height - 1)), imageChannel->width, MPI_UNSIGNED_CHAR, rank + 1, 0,
                     ghostCells->south, imageChannel->width, MPI_UNSIGNED_CHAR, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("Rank %d: Send south border to %d and recv north border from %d as new south border \n", rank, rank + 1, rank + 1);
    }
}

void exchangeNorthBorder(bmpImageChannel *imageChannel, borders *ghostCells, int rank)
{
    if (rank != 0)
    {
        // Send north border to rank-1 and recv south border from rank-1 as new north border
        MPI_Sendrecv(imageChannel->rawdata, imageChannel->width, MPI_UNSIGNED_CHAR, rank - 1, 0,
                     ghostCells->north, imageChannel->width, MPI_UNSIGNED_CHAR, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("Rank %d: Send north border to %d and recv south border from %d as new north border \n", rank, rank - 1, rank - 1);
    }
}