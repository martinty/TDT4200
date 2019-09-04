TARGET	= bitmap
CC 		= mpicc
CFLAGS = -Iinc/
RUN		= mpirun

all: main.o bitmap.o
	$(CC) $(CFLAGS) main.o bitmap.o -o $(TARGET) 

main.o: src/main.c
	$(CC) $(CFLAGS) -c src/main.c -o main.o

bitmap.o: src/bitmap.c
	$(CC) $(CFLAGS) -c src/bitmap.c -o bitmap.o

phony : clean
clean: 
	rm -f *.o
	rm -f $(TARGET)
	rm -f after.bmp 

phony : run
run:
	$(RUN) -np $(n) ./$(TARGET) 
