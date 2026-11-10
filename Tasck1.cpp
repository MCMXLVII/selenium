#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define MASTER_RANK 0
#define ARRAY_SIZE 100000000

int main(int argc, char* argv[]) {
    int rank, size;
    double start_time, end_time;

    // Инициализация MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Определяем размер локального массива для каждого MPI процесса
    int local_size = ARRAY_SIZE / size;
    int remainder = ARRAY_SIZE % size;

    // Корректировка размера для каждого процесса
    if (rank < remainder) {
        local_size++;
    }

    // Выделение памяти под локальный массив
    double* local_array = (double*)malloc(local_size * sizeof(double));
    double* full_array = NULL;

    // Только мастер-процесс создает и инициализирует полный массив
    if (rank == MASTER_RANK) {
        full_array = (double*)malloc(ARRAY_SIZE * sizeof(double));

        // Инициализация массива случайными числами
        srand(42); // Фиксированное зерно для воспроизводимости
        for (int i = 0; i < ARRAY_SIZE; i++) {
            full_array[i] = (double)(rand() % 100) / 10.0;
        }
    }

    // Распределение данных между MPI процессами
    int* send_counts = (int*)malloc(size * sizeof(int));
    int* displacements = (int*)malloc(size * sizeof(int));

    // Вычисление количества элементов для каждого процесса
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int count = ARRAY_SIZE / size;
        if (i < remainder) {
            count++;
        }
        send_counts[i] = count;
        displacements[i] = offset;
        offset += count;
    }

    // Рассылка данных
    MPI_Scatterv(full_array, send_counts, displacements, MPI_DOUBLE,
        local_array, local_size, MPI_DOUBLE,
        MASTER_RANK, MPI_COMM_WORLD);

    // Тестирование с разным количеством OpenMP потоков
    int omp_threads[] = { 1, 2, 4 };
    int num_omp_tests = 3;

    for (int test = 0; test < num_omp_tests; test++) {
        int num_threads = omp_threads[test];

        // Установка количества потоков OpenMP
        omp_set_num_threads(num_threads);

        // Синхронизация перед измерением времени
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();

        // Гибридное суммирование: OpenMP внутри MPI процесса
        double local_sum = 0.0;

#pragma omp parallel for reduction(+:local_sum)
        for (int i = 0; i < local_size; i++) {
            local_sum += local_array[i];
        }

        // Сбор результатов от всех MPI процессов
        double global_sum = 0.0;
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
            MASTER_RANK, MPI_COMM_WORLD);

        end_time = MPI_Wtime();

        // Вывод результатов (только мастер-процесс)
        if (rank == MASTER_RANK) {
            printf("MPI: %d | OpenMP: %d | Time: %.6f sec.\n",
                size, num_threads, end_time - start_time);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Вывод проверочной суммы (только мастер-процесс)
    if (rank == MASTER_RANK) {
        double check_sum = 0.0;
        for (int i = 0; i < ARRAY_SIZE; i++) {
            check_sum += full_array[i];
        }

        free(full_array);
    }

    // Очистка памяти
    free(local_array);
    free(send_counts);
    free(displacements);

    MPI_Finalize();

    return 0;
}
