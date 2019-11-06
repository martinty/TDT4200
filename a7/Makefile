PATH += :/usr/local/cuda/bin
I = 1
NVCCFLAGS += -arch=sm_61 -rdc=true -O3 -lm
DEBUGFLAGS += -lineinfo
IMAGE = before.bmp
OUTPUT = after.bmp

all: main

main:
	nvcc $(NVCCFLAGS) libs/bitmap.c main.cu -o main.out 

.PHONY: run test debug help clean

run: main
	./main.out $(IMAGE) $(OUTPUT) -i $(I)

test: main
	./main.out $(IMAGE) $(OUTPUT) -t -i $(I)

debug:
	nvcc $(NVCCFLAGS) $(DEBUGFLAGS) libs/bitmap.c main.cu -o main.out 
	cuda-memcheck ./main.out $(IMAGE) $(OUTPUT) -i $(I) |more

help: main
	./main.out -h

clean:
	rm -rf *.o
	rm -rf *.out
	rm -rf after.bmp
