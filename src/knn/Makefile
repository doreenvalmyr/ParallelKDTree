CC = mpic++
FLAGS = -std=c++14 -lpthread -Wall -g -fopenmp

# Source files
KNN_SRC = knn.cpp
KNN_PARALLEL_SRC = knn-parallel-mpi.cpp
KDTREE_SRC = ../kdTree/kdTree.cpp

# Executables
TARGET = knn.out
PARALLEL_TARGET = knn-parallel.out 

$(TARGET): $(KNN_SRC) $(KDTREE_SRC)
	$(CC) $(FLAGS) -o $@ $^

$(PARALLEL_TARGET): $(KNN_PARALLEL_SRC) $(KDTREE_SRC)
	$(CC) $(FLAGS) -o $@ $^

DEFAULT_ARGS = -k 3 -d 6 -t '1.1 1.2 1.3 1.4 1.5 1.6' -i ../datasets/small-dataset.csv
run: $(TARGET)
	./$(TARGET) $(if $(ARGS),$(ARGS),$(DEFAULT_ARGS))

run-parallel: $(PARALLEL_TARGET)
	mpirun -np 8 ./$(PARALLEL_TARGET) $(if $(ARGS),$(ARGS),$(DEFAULT_ARGS))

clean:
	rm -f $(TARGET) $(PARALLEL_TARGET)