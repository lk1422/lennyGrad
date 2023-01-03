#include <iostream>
#include <iomanip>
#include "tensor.h"

int main(){

    int arr[125];
    int dims[] = {5, 5, 5};
    for(int i=0; i<125; i++) arr[i] = i+1;
    Tensor<int> t1 (arr, 3, dims);
    t1._printInternalArr();
    for(int i=0; i<5; i++) {
        for(int j=0; j<5; j++){
            for (int k=0; k<5; k++){
                std::cout << std::setw(5) << t1.get(3, i,j,k) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << t1 << std::endl;

    std::cout << "TRANSPOSING" << std::endl;
    t1.transpose();

    t1._printInternalArr();
    for(int i=0; i<5; i++) {
        for(int j=0; j<5; j++){
            for (int k=0; k<5; k++){
                std::cout << std::setw(5) << t1.get(3, i,j,k) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << t1 << std::endl;

    t1.as_contiguous();
    std::cout << "Making contiguous" << std::endl;

    t1._printInternalArr();
    for(int i=0; i<5; i++) {
        for(int j=0; j<5; j++){
            for (int k=0; k<5; k++){
                std::cout << std::setw(5) << t1.get(3, i,j,k) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << t1 << std::endl;

    



    return 0;
}