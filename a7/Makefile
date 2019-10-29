
.PHONY: clean

main: libs/bitmap.c libs/bitmap.c main.cu
	nvcc libs/bitmap.c main.cu

clean:
	rm -Rf *.o
	rm -Rf main

# end
