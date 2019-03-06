
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <conio.h>

#include <stdlib.h>
#include <Windows.h>  // Necesario para cambiar los colores de consola en SO Windows

enum Colores {
	BLACK = 0,
	BLUE = 1,
	GREEN = 2,
	CYAN = 3,
	RED = 4,
	MAGENTA = 5,
	BROWN = 6,
	LGREY = 7,
	DGREY = 8,
	LBLUE = 9,
	LGREEN = 10,
	LCYAN = 11,
	LRED = 12,
	LMAGENTA = 13,
	YELLOW = 14,
	WHITE = 15
};

//	Cambiará el color del fondo de la consola y de fuente de lo que se escriba a continuación
void Color(int Background, int Text) {

	HANDLE Console = GetStdHandle(STD_OUTPUT_HANDLE);
	//Cálculo para convertir los colores al valor necesario
	int New_Color = Text + (Background * 16);
	//Aplicamos el color a la consola
	SetConsoleTextAttribute(Console, New_Color);

}
//--------------------------------------------------------------------------------------------------------------------------------

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
	char tecla;

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	Color(BLUE, YELLOW); printf("{1,2,3,4,5}\n");
	Color(RED, LGREEN); printf(" + {10, 20, 30, 40, 50}\n");
	Color(WHITE, BROWN); printf(" = { %d,%d,%d,%d,%d }\n", c[0], c[1], c[2], c[3], c[4]);
	Color(BLACK, WHITE);

	//--------------------------------------------------------------------------------------------------------------
	printf("\n*** Nos movemos con wasd o las flechas del teclado ***\n***                Enter para Salir                ***\n\n");

	do {
		tecla = getch();

		if (tecla == 'w' || tecla == 'W') printf("Arriba\n");
		if (tecla == 'a' || tecla == 'A') printf("Izquierda\n");
		if (tecla == 'd' || tecla == 'D') printf("Derecha\n");
		if (tecla == 's' || tecla == 'S') printf("Abajo\n");

		if (tecla == -32) {
			tecla = getch();
			if (tecla == 72) printf("Arriba\n");
			if (tecla == 75) printf("Izquierda\n");
			if (tecla == 77) printf("Derecha\n");
			if (tecla == 80) printf("Abajo\n");
		}

		printf("%c, %d\n", tecla, tecla);
	} while (tecla != 13);
	//--------------------------------------------------------------------------------------------------------------

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
