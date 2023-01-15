#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>

#include "tensor.h"
#include "iterator.h"
#include "utils.h"
#include "ops.h"
#include "grad_check.h"

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
std::vector<Tensor<T> *> readTensors(const std::string &fileName){
    std::vector<Tensor<T> *> tens;
    std::ifstream tensor_file;
    tensor_file.open(fileName);
    tensor_file.close();
}

template <typename T>
void test_func(T func, const std::string & out_file, Tensor<double> ** tensors){
    std::ifstream tensor_file;
    tensor_file.open(out_file);
    int n_tensors;
    tensor_file >> n_tensors;
    int count = 0;


    for(int i=0; i<(n_tensors*2); i+=2) {
        Tensor<double> * expected_out = getTensor(tensor_file);
        //Compute out
        /*
        std::cout << *tensors[i] << std::endl;
        std::cout << *tensors[i+1] << std::endl;
        */
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
}

int main() {
    std::ifstream tensor_file;
    tensor_file.open("testfiles/tensors.txt");
    int n_tensors;
    tensor_file >> n_tensors;

    Tensor<double> * tensors[n_tensors];
    for(int i=0; i<n_tensors; i++){
        tensors[i] = getTensor(tensor_file);
    }

    std::ifstream m_tensor_file;
    m_tensor_file.open("testfiles/m_tensors.txt");
    int m_n_tensors;
    m_tensor_file >> m_n_tensors;

    Tensor<double> * m_tensors[n_tensors];
    for(int i=0; i<m_n_tensors; i++){
        m_tensors[i] = getTensor(m_tensor_file);
    }

    std::cout << "TESTING FORWARD OPERATIONS" << std::endl;
    test_func(OPS::ADD<double>, std::string("testfiles/add.txt"), tensors);
    test_func(OPS::MULT<double>, std::string("testfiles/mult.txt"), tensors);
    test_func(OPS::MatMul<double>, std::string("testfiles/matmul.txt"), m_tensors);

    std::cout << "===========================================================" << std::endl;
    std::cout << "TESTING ADD GRADIENTS" << std::endl;
    int count = 0;
    int total_tests = 500;
    for(int i=0; i<total_tests; i++){
        Tensor<double> * tensor1 = new Tensor<double>(3,2,2,2);
        Tensor<double> * tensor2 = new Tensor<double>(3,2,2,2);
        tensor1->randn();
        tensor2->randn();
        bool passed_test = grad_check(ADDITION_TEST<double>, tensor1, tensor2);
        if(passed_test) count++;
        delete tensor1;
        delete tensor2;
    }
    std::cout << "PASSED: " << count << "/" << total_tests << std::endl;
    std::cout << "===========================================================" << std::endl;

    std::cout << "===========================================================" << std::endl;
    std::cout << "TESTING MULT GRADIENTS" << std::endl;
    count = 0;
    for(int i=0; i<total_tests; i++){
        Tensor<double> * tensor1 = new Tensor<double>(3,2,2,2);
        Tensor<double> * tensor2 = new Tensor<double>(3,2,2,2);
        tensor1->randn();
        tensor2->randn();
        bool passed_test = grad_check(MULT_TEST<double>, tensor1, tensor2);
        if(passed_test) count++;
        delete tensor1;
        delete tensor2;
    }
    std::cout << "PASSED: " << count << "/" << total_tests << std::endl;
    std::cout << "===========================================================" << std::endl;

    std::cout << "===========================================================" << std::endl;
    std::cout << "TESTING MATMULT GRADIENTS" << std::endl;
    count = 0;
    for(int i=0; i<total_tests; i++){
        Tensor<double> * tensor1 = new Tensor<double>(3,2,2,3);
        Tensor<double> * tensor2 = new Tensor<double>(3,2,3,2);
        tensor1->randn();
        tensor2->randn();
        bool passed_test = grad_check(MATMUL_TEST<double>, tensor1, tensor2);
        if(passed_test) count++;
        delete tensor1;
        delete tensor2;
    }
    std::cout << "PASSED: " << count << "/" << total_tests << std::endl;
    std::cout << "===========================================================" << std::endl;
}


