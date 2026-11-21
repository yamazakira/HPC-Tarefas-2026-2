
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double funcao(double x)
{
    return x * x;
}

int main(int argc, char *argv[])
{
    int rank, size;
    double a, b, h;
    int n;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        printf("Digite o limite inferior a: ");
        scanf("%lf", &a);

        printf("Digite o limite superior b: ");
        scanf("%lf", &b);

        printf("Digite o numero de subdivisoes n: ");
        scanf("%d", &n);

        if (n <= 0 || a >= b)
        {
            printf("Valores invalidos.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        h = (b - a) / n;
    }

    MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int inicio = rank * n / size;
    int fim = (rank + 1) * n / size;

    double soma_local = 0.0;

    for (int i = inicio; i < fim; i++)
    {
        double x1 = a + i * h;
        double x2 = a + (i + 1) * h;

        soma_local += (funcao(x1) + funcao(x2)) * h / 2.0;
    }

    if (rank == 0)
    {
        double resultado = soma_local;

        for (int p = 1; p < size; p++)
        {
            double parcial;

            MPI_Recv(&parcial, 1, MPI_DOUBLE, p, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            resultado += parcial;
        }

        double exato = (pow(b, 3) - pow(a, 3)) / 3.0;

        printf("\nIntegral aproximada (trapezio): %.10f\n",
               resultado);

        printf("Integral exata: %.10f\n", exato);

        printf("Erro: %.10f\n", fabs(resultado - exato));
    }
    else
    {
        MPI_Send(&soma_local, 1, MPI_DOUBLE, 0, 0,
                 MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
