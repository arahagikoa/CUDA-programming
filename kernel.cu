
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <memory>
#include <string>


/*
	Tutaj mamy testową klasę, której zadaniem jest trzymanie listy Int ów

*/
class Entity {
public:
	int* data;
	Entity(int size) : data(new int[size]) {}
	~Entity() { delete[] data; }
};



/*
	Funkcja globalna przechodzimy przez każdy element listy z klasy entity i mnożymy go przez 2
    Jest to funkcja CUDA która wykonuje tą operacje na GPU

*/
__global__ void kernel(int* deviceData, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        deviceData[idx] *= 2;
        printf("Thread %d: data[%d] = %d\n", idx, idx, deviceData[idx]); /* Tutaj też print żeby było widać bez debuggera co i jak */
    }
}

/* Tutaj mamy dwie podstawowe funkcje to przekazywania danych, od hosta(CPU) do GPU i w drugą stronę*/
void copyToGPU(int* device_ptr, int* host_ptr, int size) {
    cudaMemcpy(device_ptr, host_ptr, size * sizeof(int), cudaMemcpyHostToDevice);
}

/* Tutaj trzeba pamiętać aby przekazać sahred_ptr bo jeśli zmodyfikujemy dane przekazane z Entity to faktycznie Kernel wykona obliczenia, ale te dane są alokowane w innym miejscu w pamięci i faktycznie nie modyfikują
danych w sharedEntity
*/
void copyFromGPUToHost(int* device_ptr, std::shared_ptr<Entity> sharedEntity, int size) {
    cudaMemcpy(sharedEntity->data, device_ptr, size * sizeof(int), cudaMemcpyDeviceToHost);
}

int main() {
    int size = 1024;

    {
        std::shared_ptr<Entity> e0;
        {
            std::shared_ptr<Entity> sharedEntity = std::make_shared<Entity>(size);
            e0 = sharedEntity;

            
            for (int i = 0; i < size; i++) {
                sharedEntity->data[i] = i;
            }

            int* deviceData;
            cudaMalloc(&deviceData, size * sizeof(int));

            copyToGPU(deviceData, sharedEntity->data, size); 
            /* 
            this	0x000000a8f178f628 shared_ptr {data=0x000001dbec1e9af0 {0} } [2 strong refs] [make_shared]	const std::shared_ptr<Entity> *
            */

            
            /* Tutaj wiem, że podkresla tą jedną strzałkę, ale nie wiem czym to jest spowodowane, nie wygląda jakby powodowało jakieś błędy*/
            int threadsPerBlock = 256; 
            int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock; /* Tutaj mi chat podpowiedział jak to wykalkulować jak coś*/

            kernel <<<blocksPerGrid, threadsPerBlock >> > (deviceData, size); /* Nie przejmuj się tym podkreśleniem, jest to chyba wina interpretera, bo w rzeczywistośc
            nie jest to błąd */
            cudaDeviceSynchronize();
            copyFromGPUToHost(deviceData, sharedEntity, size);

            cudaFree(deviceData); /* Zwalniamy całą używaną pamięć zarezerwowaną dla naszego deviceData */

            /* Tutaj dodałem pętle for by printować te dane, jeśli nie chcesz zaglądać w debuggera */
            std::cout << "Modified Data:" << std::endl;
            for (int i = 0; i < size; i++) {
                std::cout << sharedEntity->data[i] << " ";
            }
        }
    }

    std::cin.get();
    return 0;
}