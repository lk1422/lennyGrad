#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "tensor.h"
#include "ops.h"

/*GRADIENT CHECKING*/
template <typename T, typename K>
bool grad_check(T test_func, Tensor<K> * t1, Tensor<K> * t2) {
    bool pass = true;
    const double EPS = .001;
    const double EPS_2 = EPS/2;

    Tensor<K> * out = test_func(t1,t2, true).first;//Out puts a scalar ("LOSS")
    out->getGrad()->setAll(1);
    out->getOp()->back();

    //Check Gradient
    //Check input1 gradeints.
    int arr[t1->getNDims()];
    for(int i=0; i<t1->getNDims(); i++) arr[i] = 0;
    t1->reshape_grad(t1->getNDims(), t1->getDims());
    iterator<K> it1(t1, NULL, arr);
    iterator<K> it1_g(t1->getGrad(), NULL, arr);
    for(int i=0; i<t1->getTotalElements(); i++) {
        K goal = it1_g.next();
        K& temp_ref = it1.next();
        K val = temp_ref;
        
        temp_ref += EPS_2;
        auto temp1 = test_func(t1, t2, false);
        delete temp1.first;
        K out1 = temp1.second;

        temp_ref -= 2*EPS_2;
        temp1 = test_func(t1, t2, false);
        delete temp1.first;
        K out2 = temp1.second;

        K numerical_der = (out1 - out2) / (EPS);
        if(fabs(numerical_der - goal) > .001){
            std::cout << "ERROR IN GRADIENT" << std::endl;
            std::cout << "EXPECTED GRADIENT: " << std::setprecision(10) << numerical_der << std::endl;
            std::cout << "RECIEVED GRADIENT: " << std::setprecision(10) << goal << std::endl;
            pass = false;
        }
        temp_ref = val;
    }

    //Check input2 gradients.
    int arr_2[t2->getNDims()];
    for(int i=0; i<t2->getNDims(); i++) arr_2[i] = 0;
    t2->reshape_grad(t2->getNDims(), t2->getDims());
    iterator<K> it2(t2, NULL, arr_2);
    iterator<K> it2_g(t2->getGrad(), NULL, arr);
    for(int i=0; i<t2->getTotalElements(); i++) {
        K goal = it2_g.next();
        K& temp_ref = it2.next();
        K val = temp_ref;
        
        temp_ref += EPS_2;
        auto temp1 = test_func(t1, t2, false);
        delete temp1.first;
        K out1 = temp1.second;

        temp_ref -= 2*EPS_2;
        temp1 = test_func(t1, t2, false);
        delete temp1.first;
        K out2 = temp1.second;

        K numerical_der = (out1 - out2) / (EPS);
        if(fabs(numerical_der - goal) > .001){
            std::cout << "ERROR IN GRADIENT" << std::endl;
            std::cout << "EXPECTED GRADIENT: " << std::setprecision(10) << numerical_der << std::endl;
            std::cout << "RECIEVED GRADIENT: " << std::setprecision(10) << goal << std::endl;
            pass = false;
        }

        temp_ref = val;
    }

    delete out;
    
    return pass;
}

/*GRADIENT CHECKING TEST FUNCTIONS*/
//Manually implement the sum operation
//We can implement the SUM op later and test it 
//Again using these functions
template <typename T>
std::pair<Tensor<T> *, T> ADDITION_TEST(Tensor<T>* t1, Tensor<T> * t2, bool history=true){
    if(!history) t1->no_history(); 
    Tensor<T> * out = OPS::ADD<T>(t1,t2);
    
    //Sum up all
    int arr[out->getNDims()];
    for(int i=0; i < out->getNDims(); i++) arr[i] = 0;
    T sum = 0;
    iterator<T> it1(out, NULL, arr);
    for(int i=0; i<out->getTotalElements(); i++){
        sum +=  it1.next();
    }
    return std::make_pair(out,sum);
}

template <typename T>
std::pair<Tensor<T> *, T> MULT_TEST(Tensor<T>* t1, Tensor<T> * t2, bool history=true){
    if(!history) t1->no_history(); 
    Tensor<T> * out = OPS::MULT<T>(t1,t2);
    
    //Sum up all
    int arr[out->getNDims()];
    for(int i=0; i < out->getNDims(); i++) arr[i] = 0;
    T sum = 0;
    iterator<T> it1(out, NULL, arr);
    for(int i=0; i<out->getTotalElements(); i++){
        sum +=  it1.next();
    }
    return std::make_pair(out,sum);
}

template <typename T>
std::pair<Tensor<T> *, T> MATMUL_TEST(Tensor<T>* t1, Tensor<T> * t2, bool history=true){
    if(!history) t1->no_history(); 

    Tensor<T> * out = OPS::MatMul<T>(t1,t2);
    
    //Sum up all
    int arr[out->getNDims()];
    for(int i=0; i < out->getNDims(); i++) arr[i] = 0;
    T sum = 0;
    iterator<T> it1(out, NULL, arr);
    for(int i=0; i<out->getTotalElements(); i++){
        sum +=  it1.next();
    }
    return std::make_pair(out,sum);
}