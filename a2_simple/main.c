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

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);
           

    if(world_rank == 0){
        Pixel **image = calloc(YSIZE, sizeof(Pixel *));
        for(int row = 0; row < YSIZE; row++){
            image[row] = calloc(XSIZE, sizeof(Pixel));
        }
	    
        readbmp("before.bmp", image);

	    // Alter the image here
        invertColor(image, XSIZE, YSIZE);

    	savebmp("after.bmp", image, XSIZE, YSIZE);
	    
        for(int row = 0; row < YSIZE; row++){
            free(image[row]);
        }
        free(image);
    };
    

    // Finalize the MPI environment.
    MPI_Finalize();
	

	return 0;
}