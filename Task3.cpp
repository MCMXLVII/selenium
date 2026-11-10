#include <iostream>
#include <iomanip>
#include <omp.h>
#include <ctime>
#include <locale>

using namespace std;

// Функция прямого хода метода Гаусса
void forwardGauss(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        // Нормализация строки
        double pivot = matrix[i][i];
        for (int j = i; j <= n; j++) {
            matrix[i][j] /= pivot;
        }

        // Исключение переменной - параллельная версия
#pragma omp parallel for
        for (int j = i + 1; j < n; j++) {
            double multiplier = matrix[j][i];
            for (int k = i; k <= n; k++) {
                matrix[j][k] -= multiplier * matrix[i][k];
            }
        }
    }
}

// Функция обратного хода метода Гаусса
void backwardGauss(double** matrix, int n, double* x) {
    x[n - 1] = matrix[n - 1][n];

#pragma omp parallel for
    for (int i = n - 2; i >= 0; i--) {
        x[i] = matrix[i][n];
        for (int j = i + 1; j < n; j++) {
            x[i] -= matrix[i][j] * x[j];
        }
    }
}

int main() {
    setlocale(LC_ALL, "Russian");

    int n;
    int num_threads;

    cout << "Введите размер системы (n): ";
    cin >> n;
    cout << "Введите количество потоков: ";
    cin >> num_threads;

    // Устанавливаем количество потоков
    omp_set_num_threads(num_threads);

    // Выделение памяти
    double** matrix = new double* [n];
    for (int i = 0; i < n; i++) {
        matrix[i] = new double[n + 1];
    }
    double* x = new double[n];

    // Инициализация матрицы случайными значениями
    srand(time(0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 10 + 1; // Коэффициенты от 1 до 10
        }
        matrix[i][n] = rand() % 20; // Свободные члены от 0 до 19
    }


    // Решение системы с замером времени
    clock_t start = clock();

    forwardGauss(matrix, n);
    backwardGauss(matrix, n, x);

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;


    cout << "Время выполнения: " << elapsed << " секунд\n";
    cout << "Использовано потоков: " << num_threads << "\n";


    // Освобождение памяти
    delete[] x;
    for (int i = 0; i < n; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;

    return 0;
}