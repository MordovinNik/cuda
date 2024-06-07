#pragma once

#include <vector>
#include <cuda_runtime.h>

class Matrix
{
public:
     
     Matrix(const std::size_t rows, const std::size_t cols);

     Matrix CPUMultiply(Matrix const &other) const;
     Matrix CUDAMultiply(Matrix const &other) const;
     Matrix CUDASharedMemoryMultiply(Matrix const &other) const;
     Matrix CUDAWarpIntrinsicsMultiply(Matrix const &other) const;

     virtual ~Matrix();

private:
     std::size_t _rows, _cols;
     float* _data;
};