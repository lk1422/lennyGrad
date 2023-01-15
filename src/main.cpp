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
    // Tensor<double> * t = new Tensor<double>(3, 2,2,2);
    // t->randn();
    // std::cout << *t << std::endl;
    // Tensor<double> * t_ = OPS::ReLU(t);
    // t_->init_grad();
    // t_->getGrad()->setAll(1);
    // t_->getOp()->back();


    // std::cout << *t << std::endl;
    // std::cout << *t_ << std::endl;


    // delete t;
    // delete t_;

    return 0;
}