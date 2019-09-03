#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "bitmap.h"

#define XSIZE 2560 // Size of before image
#define YSIZE 2048

int main(int argc, char** argv) {
	
	// Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Make Pixel datatype
    MPI_Datatype pixel_dt;
    MPI_Type_contiguous(3, MPI_UNSIGNED_CHAR, &pixel_dt);
    MPI_Type_commit(&pixel_dt);

    // Set size of array sent to each process
    int yScale = YSIZE / world_size;

    if(world_rank == 0){
        Pixel **image = calloc(YSIZE, sizeof(Pixel *));
        for(int row = 0; row < YSIZE; row++){
            image[row] = calloc(XSIZE, sizeof(Pixel));
        }

        readbmp("before.bmp", image);

        int rest = YSIZE % world_size;
        for(int i = 1; i < world_size; i++){
            MPI_Send(&image[yScale*i+rest][0], XSIZE*yScale, pixel_dt, i, 0, MPI_COMM_WORLD);
        }
        //invertColor(image, XSIZE, yScale + rest);
        for(int i = 1; i < world_size; i++){
            MPI_Recv(&image[yScale*i+rest][0], XSIZE*yScale, pixel_dt, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

	    savebmp("after.bmp", image, XSIZE, YSIZE);

        for(int row = 0; row < YSIZE; row++){
            free(image[row]);
        }
        free(image);
    }
    else{
        Pixel **data = calloc(yScale, sizeof(Pixel *));
        for(int row = 0; row < yScale; row++){
            data[row] = calloc(XSIZE, sizeof(Pixel));
        }

        MPI_Recv(&data[0][0], XSIZE*yScale, pixel_dt, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        invertColor(data, XSIZE, yScale);
        MPI_Send(&data[0][0], XSIZE*yScale, pixel_dt, 0, 0, MPI_COMM_WORLD);

        for(int row = 0; row < yScale; row++){
            free(data[row]);
        }
        free(data);
    }
    

    // Finalize the MPI environment.
    MPI_Finalize();
	

	return 0;
}