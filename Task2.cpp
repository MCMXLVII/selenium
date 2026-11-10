#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <ctime>
#include <cmath>
#include <locale>

using namespace std;

// Функция прямого хода метода Гаусса (MPI версия)
void forwardGaussMPI(double* matrix, int n, int rank, int size) {
    for (int i = 0; i < n; i++) {
        // Нормализация строки i (делает главный элемент равным 1)
        if (rank == 0) {
            double pivot = matrix[i * n + i];
            for (int j = i; j <= n; j++) {
                matrix[i * n + j] /= pivot;
            }
        }

        // Рассылаем нормализованную строку всем процессам
        MPI_Bcast(&matrix[i * n], n + 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Исключение переменной из остальных строк
        int start_row = (rank == 0) ? i + 1 : rank;
        for (int j = start_row; j < n; j += size) {
            if (j > i) {
                double multiplier = matrix[j * n + i];
                if (multiplier != 0) {
                    for (int k = i; k <= n; k++) {
                        matrix[j * n + k] -= multiplier * matrix[i * n + k];
                    }
                }
            }
        }

        // Сбор обновленных строк обратно на процесс 0
        if (rank != 0) {
            for (int j = rank; j < n; j += size) {
                if (j > i) {
                    MPI_Send(&matrix[j * n], n + 1, MPI_DOUBLE, 0, j, MPI_COMM_WORLD);
                }
            }
        }
        else {
            for (int p = 1; p < size; p++) {
                for (int j = p; j < n; j += size) {
                    if (j > i) {
                        MPI_Recv(&matrix[j * n], n + 1, MPI_DOUBLE, p, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
        }
    }
}

// Функция обратного хода метода Гаусса (MPI версия)
void backwardGaussMPI(double* matrix, int n, double* x, int rank, int size) {
    if (rank == 0) {
        x[n - 1] = matrix[(n - 1) * n + n];

        for (int i = n - 2; i >= 0; i--) {
            x[i] = matrix[i * n + n];
            for (int j = i + 1; j < n; j++) {
                x[i] -= matrix[i * n + j] * x[j];
            }
        }
    }

    // Рассылаем решение всем процессам
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    setlocale(LC_ALL, "Russian");
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    double start_time, end_time;

    if (rank == 0) {
        cout << "Введите размер системы (n): ";
        cin >> n;
        cout << "Количество процессов: " << size << endl;
    }

    // Рассылаем n всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Выделение памяти
    double* matrix = new double[n * (n + 1)]();
    double* x = new double[n]();

    // Инициализация матрицы случайными значениями (только на процессе 0)
    if (rank == 0) {
        srand(time(0));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i * n + j] = (rand() % 100) / 10.0 + 1.0; // Коэффициенты от 1 до 10
            }
            matrix[i * n + n] = (rand() % 200) / 10.0; // Свободные члены от 0 до 20
        }
    }

    // Рассылаем матрицу всем процессам
    MPI_Bcast(matrix, n * (n + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Синхронизация и замер времени
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Решение системы
    forwardGaussMPI(matrix, n, rank, size);

    // Обратный ход (только на процессе 0)
    backwardGaussMPI(matrix, n, x, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Вывод результатов
    if (rank == 0) {
        cout << "\nВремя выполнения MPI: " << (end_time - start_time) << " секунд\n";
    }

    // Освобождение памяти
    delete[] matrix;
    delete[] x;

    MPI_Finalize();
    return 0;
}