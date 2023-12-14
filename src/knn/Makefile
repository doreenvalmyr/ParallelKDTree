CC = mpic++
FLAGS = -std=c++14 -lpthread -Wall -g -fopenmp

# Source files
KNN_SRC = knn.cpp
KNN_MPI_SRC = knn-parallel-mpi.cpp
KNN_OPENMP_SRC = knn-parallel-openmp.cpp
KDTREE_SRC = ../kdTree/kdTree.cpp

# Executables
TARGET = knn.out
MPI_TARGET = knn-mpi.out 
OPENMP_TARGET = knn-openmp.out 

$(TARGET): $(KNN_SRC) $(KDTREE_SRC)
	$(CC) $(FLAGS) -o $@ $^

$(MPI_TARGET): $(KNN_MPI_SRC) $(KDTREE_SRC)
	$(CC) $(FLAGS) -o $@ $^

$(OPENMP_TARGET): $(KNN_OPENMP_SRC) $(KDTREE_SRC)
	$(CC) $(FLAGS) -o $@ $^

DEFAULT_ARGS = -k 10000 -d 10 -t '0 1 2 3 4 5 6 7 8 9' -i ../datasets/very-large-dataset.csv

# CHANGE DEFAULT_ARGS ex: make run-parallel ARGS="-k 100000 -d 10 -t '0 1 2 3 4 5 6 7 8 9' -i ../datasets/very-large-dataset.csv"

run: $(TARGET)
	./$(TARGET) $(if $(ARGS),$(ARGS),$(DEFAULT_ARGS))

run-mpi: $(MPI_TARGET)
	mpirun -np $(NUM_PROCS) ./$(MPI_TARGET) $(if $(ARGS),$(ARGS),$(DEFAULT_ARGS))

run-openmp: $(OPENMP_TARGET)
	./$(OPENMP_TARGET) $(if $(ARGS),$(ARGS),$(DEFAULT_ARGS))

clean:
	rm -f $(TARGET) $(MPI_TARGET) $(OPENMP_TARGET)