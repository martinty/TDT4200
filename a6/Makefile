PATH+=:/usr/local/cuda/bin
X=1

all: mandel

mandel:
	nvcc -o mandel mandel.cu -O3 -lm

run: mandel
	./mandel $(X)

clean:
	rm -f mandel
	rm -f *.bmp