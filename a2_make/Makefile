TARGET	= bitmap.out
CC 		= mpicc
CFLAGS = -Iinc/
RUN		= mpirun

all: init main.o bitmap.o
	$(CC) $(CFLAGS) obj/* -o program/$(TARGET) 

%.o: src/%.c	
	$(CC) $(CFLAGS) -c $< -o obj/$@

phony: clean init

init:
	mkdir -p obj
	mkdir -p program

clean: 
	rm -f obj/.o*
	rm -f program/*.out
	rm -f program/after.bmp 
