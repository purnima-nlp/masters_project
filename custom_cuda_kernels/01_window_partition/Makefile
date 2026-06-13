# Compiler
NVCC = nvcc

# Output executable
TARGET = window_partition

# Source files
SRC = window_partition.cu main.cpp

# Compiler flags
NVCC_FLAGS = -O2 -std=c++17

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Remove generated files
clean:
	rm -f $(TARGET)
