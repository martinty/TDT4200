
DEPS 	= bitmap.h
TARGET	= main

main : main.o bitmap.o
	mpicc main.o bitmap.o -o $(TARGET) 

# % matches anything, $< refers to the left hand side contents and $@ refers to the LHS
#Common header as dependency to force recompilation if it is modified
%.o : %.c $(DEPS)
	mpicc -c $< -o $@

phony : clean
clean: 
	rm -f *.o
	rm -f $(TARGET)
	rm -f after.bmp 

phony : run
run:
	mpirun -np $(n) ./$(TARGET) 
