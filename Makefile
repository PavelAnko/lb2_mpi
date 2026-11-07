CC      ?= mpicc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
TARGET  := pairwise_sum

.PHONY: all clean test

all: $(TARGET)

$(TARGET): src/pairwise_sum.c
	$(CC) $(CFLAGS) -o $(TARGET) src/pairwise_sum.c

clean:
	$(RM) $(TARGET)

test: $(TARGET)
	mpiexec -n 1 ./$(TARGET) 6 --demo
	mpiexec -n 4 ./$(TARGET) 6 --demo
	mpiexec -n 4 ./$(TARGET) 10007
