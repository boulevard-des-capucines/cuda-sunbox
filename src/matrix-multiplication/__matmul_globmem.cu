#include <iostream>


typedef struct _MatrixType{

    int cols;
    int rows;
    float* data;

    [[nodiscard]] size_t size_bytes() const { return rows * cols * sizeof(float); }
    [[nodiscard]] int size() const { return rows * cols; }

} Matrix;


Matrix allocateOnHost(int rows, int cols) {

    Matrix mat;

    mat.rows = rows;
    mat.cols = cols;
    mat.data = (float*) malloc(rows * cols * sizeof(float));

    return mat;
}


Matrix allocOnCuda(Matrix matOnHost) {

    Matrix mat;

    mat.rows = matOnHost.rows;
    mat.cols = matOnHost.cols;
    cudaMalloc(&mat.data, matOnHost.size_bytes());

    return mat;
}


Matrix copyToCuda(Matrix matOnHost) {

    auto mat = allocOnCuda(matOnHost);
    cudaMemcpy(mat.data, matOnHost.data, mat.size_bytes(), cudaMemcpyHostToDevice);

    return mat;
}


__global__
void MatMulKernel(const Matrix left, const Matrix right, Matrix product) {

    float accumulated_value = 0.0;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t e = 0; e < left.cols; ++e) {
        accumulated_value += left.data[row * left.cols + e]
                             * right.data[e * right.cols + col];
    }

    product.data[row * product.cols + col] = accumulated_value;
}


void invokeMatMulKernel(const Matrix left, const Matrix right, Matrix product, int block_size) {

    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid(right.cols / dimBlock.x, left.rows / dimBlock.y);

    MatMulKernel<<<dimGrid, dimBlock>>>(left, right, product);
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
    Matrix left = allocateOnHost(left_rows, left_cols);
    for (size_t i = 0; i < left.size(); ++i) {
        left.data[i] = 1.0;
    }

    int right_rows = left_cols;
    int right_cols = 16;
    Matrix right = allocateOnHost(right_rows, right_cols);
    for (size_t i = 0; i < right.size(); ++i) {
        right.data[i] = 2.0;
    }

    int product_rows = left_rows;
    int product_cols = right_cols;
    Matrix product = allocateOnHost(product_rows, product_cols);
    for (size_t i = 0; i < product.size(); ++i) {
        product.data[i] = -1.0;
    }

    MatMul(left, right, product, 16);

    // print
    for (size_t r = 0; r < product.rows; ++r) {
        for (size_t c = 0; c < product.cols; ++c) {
            std::cout << product.data[r * product.cols + c] << " ";
        }
        std::cout << std::endl;
    }
}