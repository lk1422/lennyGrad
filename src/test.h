#ifndef TESTS_H_
#define TESTS_H_

#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>

#include "tensor.h"
#include "iterator.h"
#include "utils.h"
#include "ops.h"
#include "grad_check.h"

#define PATH std::string("..")

Tensor<double> * getTensor(std::ifstream &file){
    int n_dims;
    file >> n_dims;
    int dims[n_dims];
    int total_els = 1;
    for(int i=0; i<n_dims; i++) {
        int dim;
        file >> dim;
        total_els *= dim;
        dims[i] = dim;
    }
    double data[total_els];
    for(int i=0; i<total_els; i++){
        double el;
        file >> el;
        data[i] = el;
    }

    Tensor<double> * r = new Tensor<double>(data, n_dims, dims);
    return r;
}

template <typename T>
bool test_func(T func, const std::string & out_file, Tensor<double> ** tensors){
    std::ifstream tensor_file;
    tensor_file.open(out_file);
    int n_tensors;
    tensor_file >> n_tensors;
    int count = 0;


    for(int i=0; i<(n_tensors*2); i+=2) {
        Tensor<double> * expected_out = getTensor(tensor_file);
        //Compute out
        Tensor<double> * calculated_out = func(tensors[i], tensors[i+1]);

        //Compare
        int arr[expected_out->getNDims()];
        setAllElements(expected_out->getNDims(), arr, 0);
        iterator it1(expected_out, NULL, arr);
        iterator it2(calculated_out, NULL, arr);

        bool equal = (expected_out->getNDims() == calculated_out->getNDims());
        for(int j=0; j<expected_out->getTotalElements(); j++) {
            equal &= (fabs(it1.next() - it2.next()) < .0001);
        }
        if (equal) count++;
        else{
            std::cout << "FAILED" << std::endl;
            std::cout << "EXPECTED: " << std::endl;
            std::cout << *expected_out << std::endl << std::endl;
            std::cout << *calculated_out << std::endl;
        }

        delete expected_out;
        delete calculated_out;
    }

    std::cout << "PASSED: " << count << "/" << n_tensors << " Test Cases" << std::endl;
    return count == n_tensors;
}
template <typename T>
bool testGrad(int tests,T func, bool matmul){
    int count = 0;
    for(int i=0; i<tests; i++){
        Tensor<double> * tensor1 = new Tensor<double>(3,5,4,2);
        Tensor<double> * tensor2;
        if(!matmul)
            tensor2 = new Tensor<double>(3,5,4,2);
        else {
            tensor2 = new Tensor<double>(3,5,2,4);
        }
        tensor1->randn();
        tensor2->randn();
        bool passed_test = grad_check(func, tensor1, tensor2);
        if(passed_test) count++;
        delete tensor1;
        delete tensor2;
    }
    std::cout << "PASSED: " << count << "/" << tests << std::endl;
    return count == tests;
}


bool run_tests() {
    std::ifstream tensor_file;
    tensor_file.open(PATH + "/testfiles/tensors.txt");
    int n_tensors;
    tensor_file >> n_tensors;

    Tensor<double> * tensors[n_tensors];
    for(int i=0; i<n_tensors; i++){
        tensors[i] = getTensor(tensor_file);
    }

    std::ifstream m_tensor_file;
    m_tensor_file.open(PATH + "/testfiles/m_tensors.txt");
    int m_n_tensors;
    m_tensor_file >> m_n_tensors;

    Tensor<double> * m_tensors[n_tensors];
    for(int i=0; i<m_n_tensors; i++){
        m_tensors[i] = getTensor(m_tensor_file);
    }

    bool passed_tests = true;

    std::cout << "TESTING FORWARD OPERATIONS" << std::endl;
    passed_tests &= test_func(OPS::ADD<double>, std::string(PATH + "/testfiles/add.txt"), tensors);
    passed_tests &= test_func(OPS::MULT<double>, std::string(PATH + "/testfiles/mult.txt"), tensors);
    passed_tests &= test_func(OPS::MatMul<double>, std::string(PATH + "/testfiles/matmul.txt"), m_tensors);

    int tests = 1000;
    std::cout << "===========================================================" << std::endl;
    std::cout << "TESTING ADD GRADIENTS" << std::endl;
    passed_tests &= testGrad(tests, ADDITION_TEST<double>, false);
    std::cout << "===========================================================" << std::endl;

    std::cout << "===========================================================" << std::endl;
    std::cout << "TESTING MULT GRADIENTS" << std::endl;
    passed_tests &= testGrad(tests, MULT_TEST<double>, false);
    std::cout << "===========================================================" << std::endl;

    std::cout << "===========================================================" << std::endl;
    std::cout << "TESTING MATMULT GRADIENTS" << std::endl;
    passed_tests &= testGrad(tests, MATMUL_TEST<double>, true);
    std::cout << "===========================================================" << std::endl;

    return passed_tests;
}

#endif
