all:
	nvcc -o benchmark src/main.cu

clean:
	rm -f benchmark