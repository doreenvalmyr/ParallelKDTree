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

DEFAULT_ARGS = -k 200000 -d 10 -t '0 1 2 3 4 5 6 7 8 9' -i ../datasets/very-large-dataset.csv

# CHANGE DEFAULT_ARGS ex: make run-parallel ARGS="-k 100000 -d 10 -t '0 1 2 3 4 5 6 7 8 9' -i ../datasets/very-large-dataset.csv"

run: $(TARGET)
	./$(TARGET) $(if $(ARGS),$(ARGS),$(DEFAULT_ARGS))

run-parallel: $(PARALLEL_TARGET)
	mpirun -np 8 ./$(PARALLEL_TARGET) $(if $(ARGS),$(ARGS),$(DEFAULT_ARGS))

clean:
	rm -f $(TARGET) $(PARALLEL_TARGET)