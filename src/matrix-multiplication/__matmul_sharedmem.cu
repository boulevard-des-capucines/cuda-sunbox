#include <iostream>

#define BLOCK_SIZE 16


typedef struct _MatrixType{

    int cols;
    int rows;
    int stride;
    float* data;

    [[nodiscard]] size_t size_bytes() const { return rows * stride * sizeof(float); }
    [[nodiscard]] int size() const { return rows * cols; }

} Matrix;

__device__ float GetElement(Matrix A, int row, int col) {
    return A.data[row * A.stride + col];
}

__device__ float SetElement(Matrix A, int row, int col, float value) {
    A.data[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col, int block_size) {

    Matrix sub;
    sub.rows = block_size;
    sub.cols = block_size;
    sub.stride = A.stride;
    sub.data = &A.data[row * A.stride * block_size + col * block_size];

    return sub;
}


Matrix allocateOnHost(int rows, int cols, int stride) {

    Matrix mat;

    mat.rows = rows;
    mat.cols = cols;
    mat.stride = stride;
    mat.data = (float*) malloc(rows * cols * sizeof(float));

    return mat;
}


Matrix allocOnCuda(Matrix matOnHost) {

    Matrix mat;

    mat.rows = matOnHost.rows;
    mat.cols = matOnHost.cols;
    mat.stride = matOnHost.stride;
    cudaMalloc(&mat.data, matOnHost.size_bytes());

    return mat;
}


Matrix copyToCuda(Matrix matOnHost) {

    auto mat = allocOnCuda(matOnHost);
    cudaMemcpy(mat.data, matOnHost.data, mat.size_bytes(), cudaMemcpyHostToDevice);

    return mat;
}


__global__
void MatMulKernel(const Matrix left, const Matrix right, Matrix product, int block_size) {

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix subProduct = GetSubMatrix(product, blockRow, blockCol, block_size);
    float accumulated_value = 0.0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (size_t sub_mat_i = 0; sub_mat_i < left.cols / block_size; ++sub_mat_i) {

        Matrix left_sub = GetSubMatrix(left, blockRow, sub_mat_i, block_size);
        Matrix right_sub = GetSubMatrix(right, sub_mat_i, blockCol, block_size);

        __shared__ float left_shared[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float right_shared[BLOCK_SIZE][BLOCK_SIZE];

        left_shared[row][col] = GetElement(left_sub, row, col);
        right_shared[row][col] = GetElement(right_sub, row, col);

        __syncthreads();

        for (size_t e = 0; e < block_size; ++e) {
            accumulated_value += left_shared[row][e] * right_shared[e][col];
        }

        __syncthreads();
    }

    SetElement(subProduct, row, col, accumulated_value);
}


void invokeMatMulKernel(const Matrix left, const Matrix right, Matrix product, int block_size) {

    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(right.cols / dimBlock.x, left.rows / dimBlock.y);

    MatMulKernel<<<dimGrid, dimBlock>>>(left, right, product, block_size);
}

void MatMul(Matrix left, Matrix right, Matrix product, int block_size) {

    auto d_left = copyToCuda(left);
    auto d_right = copyToCuda(right);
    auto d_product = copyToCuda(product);

    invokeMatMulKernel(d_left, d_right, d_product, block_size);

    cudaMemcpy(product.data, d_product.data, d_product.size_bytes(), cudaMemcpyDeviceToHost);

    cudaFree(d_left.data);
    cudaFree(d_right.data);
    cudaFree(d_product.data);
}


int main() {

    int left_rows = 16;
    int left_cols = 32;
    Matrix left = allocateOnHost(left_rows, left_cols, left_cols);
    for (size_t i = 0; i < left.size(); ++i) {
        left.data[i] = 1.0;
    }

    int right_rows = left_cols;
    int right_cols = 16;
    Matrix right = allocateOnHost(right_rows, right_cols, right_cols);
    for (size_t i = 0; i < right.size(); ++i) {
        right.data[i] = 2.0;
    }

    int product_rows = left_rows;
    int product_cols = right_cols;
    Matrix product = allocateOnHost(product_rows, product_cols, product_cols);
    for (size_t i = 0; i < product.size(); ++i) {
        product.data[i] = -1.0;
    }

    MatMul(left, right, product, BLOCK_SIZE);

    // print
    for (size_t r = 0; r < product.rows; ++r) {
        for (size_t c = 0; c < product.cols; ++c) {
            std::cout << product.data[r * product.cols + c] << " ";
        }
        std::cout << std::endl;
    }
}