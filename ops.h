#ifndef OPS_H_
#define OPS_H_

#include <iostream>
#include "tensor.h"

template <typename T>
struct TensorHistory{
    /*
        TensorHistory is to store the 
        original shape of the tensor when
        the operation was preformed on it
        just in case it was reshaped later on
    */
    TensorHistory(Tensor<T> * tensor, int * shape, int n_dims){
        this->shape = new int[n_dims];
        this->tensor = tensor;
        this->n_dims = n_dims;
        //Copy over old shape in case it gets changed by 
        //The original tensor
        for(int i=0; i<n_dims; i++) this->shape[i] = shape[i];
    }
    ~TensorHistory() {delete[] shape;}
    int * shape;
    int n_dims;
    Tensor<T> * tensor;
}

template <typename T>
class Op{
    public:
        Op(Tensor<T> * output, int n_in, Tensor<T>** inputs);
        ~Op();
        virtual void back(const Tensor<T> * grad) = 0;
    private:
        int n_in;
        TensorHistory<T> ** inputs;
        TensorHistory<T> * output;
        Tensor<T> ** local_grads;
};
template <typename T>
Op<T>::Op(Tensor<T> * output, int n_in, ...) {
    //Read in the inputs
    Tensor<T>* arr[n_in];
    va_list ins;
    va_start(ins, n_in);
    for(int i=0; i<n_in; i++) arr[i] = va_arg(ins, Tensor<T>*);
    //Initialize all memory
    this->n_in = n_in;
    //Create output history
    this->inputs = new TensorHistory<T>* [n_in];
    this->output = new TensorHistory<T>(output, output->shape.first, output->getNDims());
    for(int i=0; i<n_in; i++) {
        inputs[i]->init_grad();
        this->inputs[i] = new TensorHistory<T>(inputs[i], inputs[i]->shape.first, inputs[i]->getNDims());
    }
    local_grads = new Tensor<T>*[n_in];
    //Set default value to NULL
    //This will be updated after Op.back() is called
    for(int i=0; i<n_in; i++) local_grads[i] = NULL;
}
template <typename T>
Tensor<T>::~Op() {
    for(int i=0; i<n_inputs; i++){
        delete inputs[i];
        delete local_grads[i];
    }
    delete inputs;
    delete local_grads;
    delete output;
}

/********************************************************************************************/
/*                                          INDIVIAUAL OPS                                  */
/********************************************************************************************/
template <typename T>
class ADD: public Op<T>{
    ADD(Tensor<T>*output, Tensor<T>* input1, Tensor<T>* input2): Op(output, 2, input1, input2) {}
    Tesnor<T> * call(Tensor<T>* input1, Tensor<T> * input2);
};

#endif