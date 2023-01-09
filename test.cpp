#include <iostream>
#include <iomanip>
#include "tensor.h"
#include "ops.h"

int main(){

    /*
    float arr[125];
    int dims[] = {5, 5, 5};
    for(int i=0; i<125; i++) arr[i] = i+1;
    Tensor<float> * t1 = new Tensor<float>(arr, 3, dims);
    Tensor<float> * t2 = new Tensor<float>(*t1);

    Tensor<float> * t3 = OPS::ADD<float>(t1, t2);

    Tensor<float> * t4 = OPS::MULT<float> (t2, t3);

    //t1->reshape(2, 5, 25);
    t4->reshape(2, 5, 25);


    t4->init_grad();
    t4->getGrad()->setAll(1);
    t4->getOp()->back();
    t3->getOp()->back();

    std::cout << "T1: " << std::endl;
    std::cout << *t1 << std::endl;
    std::cout << "============================\n============================\n" << std::endl;
    std::cout << "T2: " << std::endl;
    std::cout << *t2 << std::endl;
    std::cout << "============================\n============================\n" << std::endl;
    std::cout << "T3: " << std::endl;
    std::cout << *t3 << std::endl;
    std::cout << "============================\n============================\n" << std::endl;
    std::cout << "T4: " << std::endl;
    std::cout << *t4 << std::endl;
    std::cout << "============================\n============================\n" << std::endl;

    std::cout << "TESTING ITERATORS" << std::endl;
    int ord[] = {2 , 0 , 1};
    iterator<float> it(t1, ord , 0, 0, 0);
    for(int i=0; i<t1->getTotalElements(); i++){
        std::cout << it.next() << std::endl;
    }

    delete t1;
    delete t2;
    delete t3;
    delete t4;
    */
    int arr1[8];
    for(int i=0; i<8; i++) arr1[i] = i+1;
    int dims[] = {2, 2, 2};
    Tensor<int> * t1 = new Tensor(arr1, 3, dims);

    int arr2[12];
    for(int i=0; i<6; i++) arr2[i] = (i+1) * 2;
    for(int i=6; i<12; i++) arr2[i] = (i-6) * 2 + 1;
    int dims2[] = {2, 2, 3};
    Tensor<int> * t2 = new Tensor(arr2, 3, dims2);
    t1->no_history();
    t2->no_history();

    std::cout << *t1 << std::endl;
    std::cout << *t2 << std::endl;
    

    int ind1[] = {0,0,0};
    int ind2[] = {0,0,0};
    int out = OPS::_dot<int>(t1,t2,ind1,ind2);

    std::cout << "OUTPUT: " << std::endl;
    std::cout << out << std::endl;
}
