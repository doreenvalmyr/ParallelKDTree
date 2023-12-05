# Compiler
CC = g++

# Compiler flags
CFLAGS = -Wall -g -fopenmp

# Source files
COMMON_SRCS = main.cpp
KD_SRCS = kdTree.cpp
KD_PARALLEL_SRCS = kdTree-parallel.cpp

# Object files
COMMON_OBJS = $(COMMON_SRCS:.cpp=.o)
KD_OBJS = $(KD_SRCS:.cpp=.o)
KD_PARALLEL_OBJS = $(KD_PARALLEL_SRCS:.cpp=.o)

# Executables
KD_TARGET = kdTree.out
KD_PARALLEL_TARGET = kdTree-parallel.out

# Default rule
all: $(KD_TARGET) $(KD_PARALLEL_TARGET)

# Rule to build kdTree executable
$(KD_TARGET): $(COMMON_OBJS) $(KD_OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Rule to build kdTree-parallel executable
$(KD_PARALLEL_TARGET): $(COMMON_OBJS) $(KD_PARALLEL_OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Rule to build object files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

DEFAULT_ARGS = -k 9 -i ../datasets/large-dataset.csv

# Run kdTree with specified arguments
run-kd: $(KD_TARGET)
	./$(KD_TARGET) $(if $(ARGS),$(ARGS),$(DEFAULT_ARGS))

# Run kdTree-parallel with specified arguments
run-kd-parallel: $(KD_PARALLEL_TARGET)
	./$(KD_PARALLEL_TARGET) $(if $(ARGS),$(ARGS),$(DEFAULT_ARGS))

# Clean rule
clean:
	rm -f $(COMMON_OBJS) $(KD_OBJS) $(KD_PARALLEL_OBJS) $(KD_TARGET) $(KD_PARALLEL_TARGET)
