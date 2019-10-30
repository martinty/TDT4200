PATH+=:/usr/local/cuda/bin
I=1

.PHONY: clean run

all: main

main:
	nvcc -o main.out libs/bitmap.c main.cu -O3 -lm

run: main
	./main.out before.bmp after.bmp -i $(I)

clean:
	rm -rf *.o
	rm -rf *.out
	rm -rf after.bmp
