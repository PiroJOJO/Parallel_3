#include <iostream>
#include <algorithm>
#include <chrono>
#include <cublas_v2.h>
#include <openacc.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i)) //x, y  

//Функция, которая высчитывает интервалы между точками внутри сетки
#pragma acc routine seq
double steps(size_t n, size_t step, double left, double right)
{
    double val = (right - left)/(n - 1);
    return left +  val * step;
}

//Функция для просмотра матрицы
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
    double error = std::stod(argv[2]);//Значение ошибки
    size_t iter = std::stoi(argv[4]);//Количество итераций
    size_t n = std::stoi(argv[6]);//Размер сетки 
    //Объявляем необходимы еперменные 
    double* vec = new double[n*n];//Массив для значений на предыдущем шаге
    double* new_vec = new double[n*n];//Массив для значений на текущем шаге
    double* tmp = new double[n*n];//Вспомогаетльный массив для сохранения результата для следующей итерации 
    double max_error = error + 1; //Объявление максимальной ошибки 
    size_t it = 0;//Счетчик итераций
    const double a = -1;//Коэффициент для вычисления разницы между двумя матрицами 
    //Создаем переменные для работы с cuBlas
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);//Использование адресов на графическом процессоре
    int max_idx = 0;//Индекс максимальной ошибки в матрице

    //Заполнение угловых значений
    vec[IDX2C(0, 0, n)] = 10;
    vec[IDX2C(n - 1, 0, n)]= 20;
    vec[IDX2C(n - 1, n - 1, n)] = 30;
    vec[IDX2C(0, n - 1, n)] = 20;

#pragma acc data copyout(new_vec[0:n*n], tmp[0:n*n]) copy(vec[0:n*n], max_error, max_idx, a) //Переносим занчения на видеокарту
    {
        //Заполнение рамок матриц
#pragma acc parallel loop independent//Создание ядра для распарреллеливания цикла
        for (size_t i = 1; i < n - 1; ++i) 
        {
            vec[IDX2C(i, 0, n)] = steps(n, i, vec[IDX2C(0, 0, n)],
                                              vec[IDX2C(n - 1, 0, n)]); //Заполнение значениями для первой строки матрицы
            vec[IDX2C(i, n - 1, n)] = steps(n, i, vec[IDX2C(0, n - 1, n)],
                                                  vec[IDX2C(n - 1, n - 1, n)]);//Заполнение значениями для последней строки матрицы
            vec[IDX2C(0, i, n)] = steps(n, i, vec[IDX2C(0, 0, n)],
                                              vec[IDX2C(0, n - 1, n)]);//Заполнение значениями превого столбца матрицы
            vec[IDX2C(n - 1, i, n)] = steps(n, i, vec[IDX2C(n - 1, 0, n)],
                                                  vec[IDX2C(n - 1, n - 1, n)]);//Заполнение значениями последнего столбца матрицы
        }
//Копирование основного вектора в дополнительные
#pragma acc parallel loop collapse(2) //Создание ядра для распарраллеливания двойоного цикла
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                new_vec[IDX2C(j, i, n)] = vec[IDX2C(j, i, n)];
                tmp[IDX2C(j, i, n)] = vec[IDX2C(j, i, n)];
            }
        }
//Цикл основного алгоритма 
        while(error < max_error && it < iter)
	    { 
#pragma acc kernels //Использование переменной на графический процессор   
            max_error = 0;          
            it++;
#pragma acc parallel loop collapse(2) present(new_vec[:n*n], vec[:n*n]) //Распараллеливание двойного цикла 
            for (int j = 1; j < n - 1; ++j)
            {
                for (int k = 1; k < n - 1; ++k)
                {
                    new_vec[IDX2C(k, j, n)] = 0.25 * (vec[IDX2C(k - 1, j, n)] + vec[IDX2C(k + 1, j, n)]
                                        + vec[IDX2C(k, j - 1, n)] + vec[IDX2C(k, j + 1, n)]);//Вычисление среднего соседий для каждой ячейки матрицы
                }
            }

// #pragma acc parallel loop collapse(2) independent
// 	        for (int j = 1; j < n - 1; ++j) 
//                 for (int k = 1; k < n - 1; ++k) 
//                     vec[IDX2C(k, j, n)] = new_vec[IDX2C(k, j, n)];
//         acc_memcpy_device(acc_deviceptr(vec), acc_deviceptr(new_vec), n * n * sizeof(double));


//Обмен между массивами с старыми значенями и с новыми через указатель
        double* swap = vec;
        vec = new_vec;
        new_vec = swap;

        acc_attach((void**)vec);
        acc_attach((void**)new_vec);
//Реализация редукции с помощью блиблиотеки cuBlas
#pragma acc data present(tmp[:n*n], vec[:n*n], new_vec[:n*n], max_idx, a)
                {
#pragma acc host_data use_device(new_vec, vec, tmp, max_idx, a) //Передача адреса переменных на графическом процессоре 
                    {
                    cublasDaxpy(handle, n*n, &a, new_vec, 1, tmp, 1);//Вычисляем разницу между новой и старой матрицей
                                        // if (status != CUBLAS_STATUS_SUCCESS) 
										// 	        std::cout<<"failed_1";
                    cublasIdamax(handle, n*n, tmp, 1, &max_idx);//Берем индекс максимального элемента разницы
                                        // if (status != CUBLAS_STATUS_SUCCESS) 
										// 	        std::cout<<"failed_2";
                    #pragma acc kernels
                        max_error = std::abs(tmp[max_idx - 1]);//Обновляем значение максимальной ошибки по индексу

                    cublasDcopy(handle, n*n, new_vec, 1, tmp, 1);//Сохраняем наши новые значения в вспомогательный массив 
                                        // if (status != CUBLAS_STATUS_SUCCESS) 
										// 	         std::cout<<"failed_3";
                    }
#pragma acc update host(max_error)//Обновляем занчения на CPU
            }
        }
    }

    std::cout<<"Error: "<<max_error<<std::endl;
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end-begin);
    std::cout<<"time: "<<elapsed_ms.count()<<" mcs\n";
    std::cout<<"Iterations: "<<it<<std::endl;

    print_matrix(vec, n);
    delete [] tmp; 
    delete [] vec; 
    delete [] new_vec;
    cublasDestroy(handle);
    return 0;  
}