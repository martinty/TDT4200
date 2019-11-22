PATH += :/usr/local/cuda/bin
NVCCFLAGS += -O3 -lm
DEBUGFLAGS += -lineinfo -Xptxas="-v"

all: main

main:
	nvcc $(NVCCFLAGS) main.cu -o main.out 

.PHONY: run time debug clean

run: main
	./main.out

time: main
	/usr/bin/time ./main.out

debug:
	nvcc $(NVCCFLAGS) $(DEBUGFLAGS) main.cu -o main.out 
	cuda-memcheck ./main.out |more

clean:
	rm -rf *.o
	rm -rf *.out
