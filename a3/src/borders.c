#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "bitmap.h"
#include "borders.h"

void exchangeHorizontalBorders(bmpImageChannel *imageChannel, int ghostRows, int rank, int size)
{
    if (rank % 2 == 0)
    {
        exchangeSouthBorder(imageChannel, ghostRows, rank, size);
        exchangeNorthBorder(imageChannel, ghostRows, rank);
    }
    else
    {
        exchangeNorthBorder(imageChannel, ghostRows, rank);
        exchangeSouthBorder(imageChannel, ghostRows, rank, size);
    }
}

void exchangeSouthBorder(bmpImageChannel *imageChannel, int ghostRows, int rank, int size)
{
    if (rank != size - 1)
    {
        int sendOffset = imageChannel->width * (imageChannel->height - ghostRows * 2);
        int recvOffset = imageChannel->width * (imageChannel->height - ghostRows);
        int count = imageChannel->width * ghostRows;

        // Send south border to rank+1 and recv north border from rank+1 as new south border (ghost rows)
        MPI_Sendrecv(imageChannel->rawdata + sendOffset, count, MPI_UNSIGNED_CHAR, rank + 1, 0,
                     imageChannel->rawdata + recvOffset, count, MPI_UNSIGNED_CHAR, rank + 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        //printf("Rank %d: Send south border to %d and recv north border from %d as new south border (ghost rows) \n", rank, rank + 1, rank + 1);
    }
}

void exchangeNorthBorder(bmpImageChannel *imageChannel, int ghostRows, int rank)
{
    if (rank != 0)
    {
        int sendOffset = imageChannel->width * ghostRows;
        int recvOffset = 0;
        int count = imageChannel->width * ghostRows;

        // Send north border to rank-1 and recv south border from rank-1 as new north border (ghost rows)
        MPI_Sendrecv(imageChannel->rawdata + sendOffset, count, MPI_UNSIGNED_CHAR, rank - 1, 0,
                     imageChannel->rawdata + recvOffset, count, MPI_UNSIGNED_CHAR, rank - 1, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                     
        //printf("Rank %d: Send north border to %d and recv south border from %d as new north border (ghost rows) \n", rank, rank - 1, rank - 1);
    }
}