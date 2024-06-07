#include "Matrix.cuh"

#include <iostream>
#include <chrono>

void main()
{
	size_t size = 1024;

	Matrix m1 = Matrix::createRandomMatrix(size, size, -100, 100);
	Matrix m2 = Matrix::createRandomMatrix(size, size, -100, 100);


	auto start = std::chrono::steady_clock::now();
	Matrix result1 = m1.CPUMultiply(m2);
	auto end = std::chrono::steady_clock::now();
	auto dif = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "CPU multiplication time: " << dif << std::endl;

	start = std::chrono::steady_clock::now();
	Matrix result2 = m1.CUDAMultiply(m2);
	end = std::chrono::steady_clock::now();
	dif = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "CUDA base multiplication time: " << dif << std::endl;

	start = std::chrono::steady_clock::now();
	Matrix result2 = m1.CUDASharedMemoryMultiply(m2);
	end = std::chrono::steady_clock::now();
	dif = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "CUDA with shared memory multiplication time: " << dif << std::endl;

	start = std::chrono::steady_clock::now();
	Matrix result2 = m1.CUDAWarpIntrinsicsMultiply(m2);
	end = std::chrono::steady_clock::now();
	dif = std::chrono::duration<double, std::milli>(end - start).count();
	std::cout << "CUDA intrinsics multiplication time: " << dif << std::endl;

	CUDA_CHECK_ERROR(cudaDeviceReset());
}