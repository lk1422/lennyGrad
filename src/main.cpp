#include <iostream>
#include <iomanip>
#include "tensor.h"
#include "ops.h"
#include "grad_check.h"
#include "test.h"

int main(){
    bool passed_all = run_tests();
    if(passed_all)
        std::cout << "PASSED ALL TEST CASES" << std::endl;
    else{
        std::cout << "FAILED TESTS" << std::endl;
        exit(-1);
    }

    return 0;
}