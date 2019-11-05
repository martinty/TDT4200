PATH += :/usr/local/cuda/bin
I = 1
NVCCFLAGS += -arch=sm_61 -rdc=true -O3 -lm
DEBUGFLAGS += -lineinfo

all: main

main:
	nvcc $(NVCCFLAGS) libs/bitmap.c main.cu -o main.out 

.PHONY: run test debug help clean

run: main
	./main.out before.bmp after.bmp -i $(I)

test: main
	./main.out before.bmp after.bmp -t -i $(I)

debug:
	nvcc $(NVCCFLAGS) $(DEBUGFLAGS) libs/bitmap.c main.cu -o main.out 
	cuda-memcheck ./main.out before.bmp after.bmp -i $(I) |more

help: main
	./main.out -h

clean:
	rm -rf *.o
	rm -rf *.out
	rm -rf after.bmp
