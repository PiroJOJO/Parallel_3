#include <iostream>
#include <algorithm>
#include <chrono>
#include <cublas_v2.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i)) //x, y  

#pragma acc routine seq
inline double steps(size_t n, size_t step, double left, double right)
{
    double val = (right - left)/(n - 1);
    return left +  val * step;
}

void print_matrix(double* vec, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            std::cout<<vec[IDX2C(j, i, n)]<<' ';
        }
        std::cout<<std::endl;
    }
}

int main(int argc, char *argv[]) {
    auto begin = std::chrono::steady_clock::now();
    if (argc != 7)
    {
        std::cout<<"Enter a string like this: Accuracy _ iterations _ size _"<<std::endl;
    }

    //Считывание значений с командной строки
    double error = std::stod(argv[2]);
    size_t iter = std::stoi(argv[4]);
    size_t n = std::stoi(argv[6]);
    double* vec = new double[n*n];
    double* new_vec = new double[n*n];
    double* tmp = new double[n*n];
    size_t it = 0;
    double negOne = -1;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t status;
    int max_idx = 0;

    //Заполнение угловых значений
    vec[IDX2C(0, 0, n)] = 10;
    vec[IDX2C(n - 1, 0, n)]= 20;
    vec[IDX2C(n - 1, n - 1, n)] = 30;
    vec[IDX2C(0, n - 1, n)] = 20;

    //Заполнение рамок матриц
#pragma acc data copy(vec[0:n*n], new_vec[0:n*n], tmp[0:n*n])
    {
#pragma acc parallel loop
        for (size_t i = 1; i < n - 1; ++i) 
        {
            vec[IDX2C(i, 0, n)] = steps(n, i, vec[IDX2C(0, 0, n)],
                                              vec[IDX2C(n - 1, 0, n)]);
            vec[IDX2C(i, n - 1, n)] = steps(n, i, vec[IDX2C(0, n - 1, n)],
                                                  vec[IDX2C(n - 1, n - 1, n)]);
            vec[IDX2C(0, i, n)] = steps(n, i, vec[IDX2C(0, 0, n)],
                                              vec[IDX2C(0, n - 1, n)]);
            vec[IDX2C(n - 1, i, n)] = steps(n, i, vec[IDX2C(n - 1, 0, n)],
                                                  vec[IDX2C(n - 1, n - 1, n)]);
        }
        std::copy_n(vec, n * n, new_vec);
        double max_error = error + 1;             
        while(error < max_error && it < iter)
	    { 
                max_error = 0;
                it ++;
#pragma acc data copy(max_error)              
#pragma acc parallel loop collapse(2) independent reduction(max:max_error)
                for (int j = 1; j < n - 1; ++j)
                {
                    for (int k = 1; k < n - 1; ++k)
                    {
                        new_vec[IDX2C(k, j, n)] =
                                0.25 * (vec[IDX2C(k - 1, j, n)] + vec[IDX2C(k + 1, j, n)]
                                        + vec[IDX2C(k, j - 1, n)] + vec[IDX2C(k, j + 1, n)]);
                        // max_error = std::max(max_error,
                        //                      std::abs(get_element(vec, n, k, j) - get_element(new_vec, n, k, j)));
                    }
                }
                #pragma acc parallel loop collapse(2)
	                for (int j = 1; j < n - 1; ++j) 
                        for (int k = 1; k < n - 1; ++k) 
                            vec[IDX2C(k, j, n)] = new_vec[IDX2C(k, j, n)];
                #pragma acc data present(tmp[:n*n], vec[:n*n], new_vec[:n*n]) wait
                {
                    #pragma acc host_data use_device(new_vec, vec, tmp)
                    {

                        status = cublasDcopy(handle, n*n, vec, 1, tmp, 1);
                        if(status != CUBLAS_STATUS_SUCCESS) std::cout << "copy error" << std::endl, exit(30);

                        status = cublasDaxpy(handle, n*n, &negOne, new_vec, 1, tmp, 1);
                        if(status != CUBLAS_STATUS_SUCCESS) std::cout << "sum error" << std::endl, exit(40);
                    
                        status = cublasIdamax(handle, n*n, tmp, 1, &max_idx);
                        if(status != CUBLAS_STATUS_SUCCESS) std::cout << "abs max error" << std::endl, exit(41);
                    }
                }

                #pragma acc update self(tmp[max_idx-1])
                error = fabs(tmp[max_idx-1]);

            }
            std::cout<<"Error: "<<max_error<<std::endl;
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin);
    std::cout<<"time: "<<elapsed_ms.count()<<" mcs\n";
    std::cout<<"Iterations: "<<it<<std::endl;
    print_matrix(vec, n);
    delete [] vec;
    delete [] new_vec;
    cublasDestroy(handle);
    return 0;
}