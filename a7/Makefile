PATH+=:/usr/local/cuda/bin
I=1

all: main

main:
	nvcc -o main.out libs/bitmap.c main.cu -O3 -lm

.PHONY: run test debug help clean

run: main
	./main.out before.bmp after.bmp -i $(I)

test: main
	./main.out before.bmp after.bmp -t -i $(I)

debug:
	nvcc -lineinfo -o main.out libs/bitmap.c main.cu -O3 -lm
	cuda-memcheck ./main.out before.bmp after.bmp -i $(I) |more

help: main
	./main.out -h

clean:
	rm -rf *.o
	rm -rf *.out
	rm -rf after.bmp
