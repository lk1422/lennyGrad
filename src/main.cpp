#include <iostream>
#include <iomanip>
#include "tensor.h"
#include "ops.h"
#include "grad_check.h"
#include "test.h"

#define DEBUG true

int main(){
    if(DEBUG){
        bool passed_all = run_tests();
        if(passed_all)
            std::cout << "PASSED ALL TEST CASES" << std::endl;
        else{
            std::cout << "FAILED TESTS" << std::endl;
            exit(-1);
        }
    }

    // Tensor<double> * X = new Tensor<double>(2, 5,5);
    // X->randn();
    // Tensor<double> * out = OPS::PAD<double>(X);
    // out->getGrad()->setAll(1);
    // out->getOp()->back();

    // std::cout << *X << std::endl;
    // std::cout << *out << std::endl;
    
    // delete out;
    return 0;
}
    // // double data[] = {1,3,5,7,9,11,13,15,17, 0,1,2,3,4,5,6,7,8};
    // // int dims[] = {1,2,3,3};
    // // Tensor<double> * X = new Tensor<double>(data, 4, dims);

    // // double kernel[] = {0.5,0,0,0.5, 1,2,1,2, 1,0,1,0, 0,0,1,1};
    // // int dims2[] = {2,2,2,2};
    // // Tensor<double> * K = new Tensor<double>(kernel, 4, dims2);

    // Tensor<double> * X = new Tensor<double>(4, 32, 3, 128, 128);
    // Tensor<double> * K = new Tensor<double>(4, 32,3,3,3);
    // X->randn();
    // K->randn();

    // //std::cout << *X << std::endl;
    // //std::cout << *K << std::endl;

    // Tensor<double> * out = conv(X, K, std::make_pair(2,2));
    // //std::cout << *out << std::endl;
    // delete X;
    // delete K;
    // delete out;