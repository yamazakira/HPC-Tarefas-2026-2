#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double funcao(double x)
{
    return x * x;
}

double calcular_passo(double a, double b, int n)
{
    return (b - a) / n;
}

double calcular_soma(double a, double h, int n)
{
    double soma = 0.0;

    for (int i = 1; i < n; i++)
    {
        double x = a + i * h;
        soma += funcao(x);
    }

    return soma;
}

double calcular_integral(double a, double b, int n)
{
    double h = calcular_passo(a, b, n);

    double soma = calcular_soma(a, h, n);

    return h * (funcao(a) / 2.0 + funcao(b) / 2.0 + soma);
}

int main(void)
{
    double a;
    double b;
    int n;

    printf("Digite o limite inferior a: ");
    scanf("%lf", &a);

    printf("Digite o limite superior b: ");
    scanf("%lf", &b);

    printf("Digite o numero de subdivisoes n: ");
    scanf("%d", &n);

    if (n <= 0 || a >= b)
    {
        printf("Valores invalidos.\n");
        return EXIT_FAILURE;
    }

    double resultado = calcular_integral(a, b, n);
    // formula da integral exata pra x^2
    double exato = (pow(b, 3) - pow(a, 3)) / 3.0;

    printf("\nIntegral aproximada (trapezio): %.10f\n", resultado);
    printf("Integral exata: %.10f\n", exato);
    printf("Erro: %.10f\n", fabs(resultado - exato));

    return EXIT_SUCCESS;
}