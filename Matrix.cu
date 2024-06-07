#include "matrix.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK_ERROR(err) if (err != cudaSuccess) {printf("Cuda error: %s\n", cudaGetErrorString(err));printf("Error in file: %s, line: %i\n", __FILE__, __LINE__);}

const size_t WARP_SIZE = 32;

Matrix::~Matrix()
{
	delete[] _data;
}

Matrix::Matrix(size_t rows, size_t cols)
	: _data(nullptr), _rows(rows), _cols(cols)
{
	_data = new float[rows * cols];
}

Matrix::Matrix(Matrix const& other) : Matrix(other._rows, other._cols)
{
	memcpy(_data, other._data, _rows * _cols * sizeof(float));
}

Matrix& Matrix::operator=(Matrix const& other)
{
	delete[] _data;
	_rows = other._rows;
	_cols = other._cols;
	_data = new float[_rows * _cols];
	memcpy(_data, other._data, _rows * _cols * sizeof(float));

	return *this;
}


__global__ void simpleMatMul(float* a, float* b, float* c, size_t l, size_t m, size_t n)
{
	size_t cCol = blockIdx.x * blockDim.x + threadIdx.x;
	size_t cRow = blockIdx.y * blockDim.y + threadIdx.y;

	if (cRow >= l || cCol >= n)
		return;

	float sum = 0;
	for (size_t i = 0; i < m; i++)
		sum += a[cRow * m + i] * b[i * n + cCol];
	c[cRow * n + cCol] = sum;
}

__global__ void sharedMatMul(float* a, float* b, float* c, size_t l, size_t m, size_t n)
{
	size_t cCol = blockIdx.x * blockDim.x + threadIdx.x;
	size_t cRow = blockIdx.y * blockDim.y + threadIdx.y;

	size_t tileCol = threadIdx.x;
	size_t tileRow = threadIdx.y;


	__shared__ float aTile[WARP_SIZE][WARP_SIZE];
	__shared__ float bTile[WARP_SIZE][WARP_SIZE + 1];

	float cVal = 0.f;
	bool isOutOfC = cRow >= l || cCol >= n;

	for (size_t tileId = 0; tileId < (m - 1) / WARP_SIZE + 1; tileId++)
	{

		aTile[tileRow][tileCol] = !isOutOfC ? a[cRow * m + (tileId * WARP_SIZE + tileCol)] : 0.f;
		bTile[tileRow][tileCol] = !isOutOfC ? b[(tileId * WARP_SIZE + tileRow) * n + cCol] : 0.f;
		__syncthreads();


		for (size_t i = 0; i < WARP_SIZE; i++)
			cVal += aTile[tileRow][i] * bTile[i][tileCol];
		__syncthreads();
	}
	if (!isOutOfC)
		c[cRow * n + cCol] = cVal;
}

__global__ void warpIntrinsicsMatMul(float* a, float* b, float* c, size_t l, size_t m, size_t n)
{
	size_t cCol = blockIdx.x * blockDim.x + threadIdx.x;
	size_t cRow = blockIdx.y * blockDim.y + threadIdx.y;
	size_t tileCol = threadIdx.x;
	size_t tileRow = threadIdx.y;

	__shared__ float aTile[WARP_SIZE][WARP_SIZE];
	__shared__ float bTile[WARP_SIZE][WARP_SIZE + 1];

	float cVal = 0.f;
	bool isOutOfC = cRow >= l || cCol >= n;

	for (size_t tileId = 0; tileId < (m - 1) / WARP_SIZE + 1; tileId++)
	{
		aTile[tileRow][tileCol] = !isOutOfC ? a[cRow * m + (tileId * WARP_SIZE + tileCol)] : 0.f;
		bTile[tileRow][tileCol] = !isOutOfC ? b[(tileId * WARP_SIZE + tileRow) * n + cCol] : 0.f;
		__syncthreads();

		float aTileLocal = aTile[tileRow][tileCol];
		__syncwarp();
		for (size_t i = 0; i < WARP_SIZE; i++)
			cVal += __shfl_sync(0xffffffff, aTileLocal, i) * bTile[i][tileCol];
		__syncthreads();
	}
	if (!isOutOfC)
		c[cRow * n + cCol] = cVal;
}

void matrixMultiplication(float const* a, float const* b, float* c, size_t l, size_t m, size_t n, Matrix::MulMode mode)
{
	CUDA_CHECK_ERROR(cudaSetDevice(0));
	CudaPtr<float> aDev(nullptr), bDev(nullptr), cDev(nullptr);
	CUDA_CHECK_ERROR(cudaMalloc(&aDev, l * m * sizeof(float)));
	CUDA_CHECK_ERROR(cudaMalloc(&bDev, m * n * sizeof(float)));
	CUDA_CHECK_ERROR(cudaMalloc(&cDev, l * n * sizeof(float)));

	CUDA_CHECK_ERROR(cudaMemcpy(aDev.get(), a, l * m * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(bDev.get(), b, m * n * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockInGrid((n - 1ULL) / WARP_SIZE + 1ULL, (l - 1ULL) / WARP_SIZE + 1ULL);
	dim3 threadInBlock(WARP_SIZE, WARP_SIZE);
	switch (mode) {
	case Matrix::MulMode::SIMPLE:
		simpleMatMul << < blockInGrid, threadInBlock >> > (aDev.get(), bDev.get(), cDev.get(), l, m, n);
		break;
	case Matrix::MulMode::SHARED:
		sharedMatMul << < blockInGrid, threadInBlock >> > (aDev.get(), bDev.get(), cDev.get(), l, m, n);
		break;
	case Matrix::MulMode::INTRINSICS:
		warpIntrinsicsMatMul << < blockInGrid, threadInBlock >> > (aDev.get(), bDev.get(), cDev.get(), l, m, n);
		break;
	}

	CUDA_FAIL(cudaDeviceSynchronize());
	CUDA_FAIL(cudaMemcpy(c, cDev.get(), l * n * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_FAIL(cudaGetLastError());
}

Matrix Matrix::CPUMultiply(Matrix const& other) const
{
	Matrix result = Matrix(_rows, other._cols);

	for (std::size_t i = 0; i < _rows; ++i)
		for (std::size_t j = 0; j < other._cols; ++j)
		{
			result._data[i * other._cols + j] = 0;
			for (std::size_t k = 0; k < _cols; ++k)
				result._data[i * other._cols + j] += _data[i * _cols + k] * other._data[k * other._cols + j];
		}
	return result;
}


Matrix Matrix::CUDAMultiply(Matrix const& other) const
{
	CUDA_CHECK_ERROR(cudaSetDevice(0));
	float* leftInput, * rightInput, * output;
	CUDA_CHECK_ERROR(cudaMalloc((void**)&leftInput, _rows * _cols * sizeof(float)));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&rightInput, _cols * other._cols * sizeof(float)));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&output, _rows * other._cols * sizeof(float)));

	CUDA_CHECK_ERROR(cudaMemcpy(leftInput, _data, _rows * _cols * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(rightInput, other._data, _cols * other._cols * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockInGrid((other._cols - 1ULL) / WARP_SIZE + 1ULL, (_rows - 1ULL) / WARP_SIZE + 1ULL);
	dim3 threadInBlock(WARP_SIZE, WARP_SIZE);

	simpleMatMul <<<blockInGrid, threadInBlock >>> (leftInput, rightInput, output, _rows, _cols, other._cols);

	Matrix result = Matrix(_rows, other._cols);

	CUDA_CHECK_ERROR(cudaDeviceSynchronize());
	CUDA_CHECK_ERROR(cudaMemcpy(result._data, output, _rows * other._cols * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERROR(cudaGetLastError());

	return result;
}

Matrix Matrix::CUDASharedMemoryMultiply(Matrix const& other) const
{
	CUDA_CHECK_ERROR(cudaSetDevice(0));
	float* leftInput, * rightInput, * output;
	CUDA_CHECK_ERROR(cudaMalloc((void**)&leftInput, _rows * _cols * sizeof(float)));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&rightInput, _cols * other._cols * sizeof(float)));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&output, _rows * other._cols * sizeof(float)));

	CUDA_CHECK_ERROR(cudaMemcpy(leftInput, _data, _rows * _cols * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(rightInput, other._data, _cols * other._cols * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockInGrid((other._cols - 1ULL) / WARP_SIZE + 1ULL, (_rows - 1ULL) / WARP_SIZE + 1ULL);
	dim3 threadInBlock(WARP_SIZE, WARP_SIZE);

	sharedMatMul << <blockInGrid, threadInBlock >> > (leftInput, rightInput, output, _rows, _cols, other._cols);

	Matrix result = Matrix(_rows, other._cols);

	CUDA_CHECK_ERROR(cudaDeviceSynchronize());
	CUDA_CHECK_ERROR(cudaMemcpy(result._data, output, _rows * other._cols * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERROR(cudaGetLastError());

	return result;
}

Matrix Matrix::CUDAWarpIntrinsicsMultiply(Matrix const& other) const
{
	CUDA_CHECK_ERROR(cudaSetDevice(0));
	float* leftInput, * rightInput, * output;
	CUDA_CHECK_ERROR(cudaMalloc((void**)&leftInput, _rows * _cols * sizeof(float)));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&rightInput, _cols * other._cols * sizeof(float)));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&output, _rows * other._cols * sizeof(float)));

	CUDA_CHECK_ERROR(cudaMemcpy(leftInput, _data, _rows * _cols * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(rightInput, other._data, _cols * other._cols * sizeof(float), cudaMemcpyHostToDevice));

	dim3 blockInGrid((other._cols - 1ULL) / WARP_SIZE + 1ULL, (_rows - 1ULL) / WARP_SIZE + 1ULL);
	dim3 threadInBlock(WARP_SIZE, WARP_SIZE);

	warpIntrinsicsMatMul << <blockInGrid, threadInBlock >> > (leftInput, rightInput, output, _rows, _cols, other._cols);

	Matrix result = Matrix(_rows, other._cols);

	CUDA_CHECK_ERROR(cudaDeviceSynchronize());
	CUDA_CHECK_ERROR(cudaMemcpy(result._data, output, _rows * other._cols * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERROR(cudaGetLastError());

	return result;
}