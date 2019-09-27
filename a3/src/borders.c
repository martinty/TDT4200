#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "bitmap.h"
#include "borders.h"

void exchangeHorizontalBorders(bmpImageChannel *imageChannel, int rank, int size)
{
    if (rank % 2 == 0)
    {
        exchangeSouthBorder(imageChannel, rank, size);
        exchangeNorthBorder(imageChannel, rank);
    }
    else
    {
        exchangeNorthBorder(imageChannel, rank);
        exchangeSouthBorder(imageChannel, rank, size);
    }
}

void exchangeSouthBorder(bmpImageChannel *imageChannel, int rank, int size)
{
    if (rank != size - 1)
    {
        // Send south border to rank+1 and recv north border from rank+1 as new south border
        MPI_Sendrecv(imageChannel->rawdata + imageChannel->width * (imageChannel->height - 2), imageChannel->width, MPI_UNSIGNED_CHAR, rank + 1, 0,
                     imageChannel->rawdata + imageChannel->width * (imageChannel->height - 1), imageChannel->width, MPI_UNSIGNED_CHAR, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("Rank %d: Send south border to %d and recv north border from %d as new south border \n", rank, rank + 1, rank + 1);
    }
}

void exchangeNorthBorder(bmpImageChannel *imageChannel, int rank)
{
    if (rank != 0)
    {
        // Send north border to rank-1 and recv south border from rank-1 as new north border
        MPI_Sendrecv(imageChannel->rawdata + imageChannel->width, imageChannel->width, MPI_UNSIGNED_CHAR, rank - 1, 0,
                     imageChannel->rawdata                      , imageChannel->width, MPI_UNSIGNED_CHAR, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("Rank %d: Send north border to %d and recv south border from %d as new north border \n", rank, rank - 1, rank - 1);
    }
}