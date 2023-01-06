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
    TensorHistory(Tensor<T> * tensor, const int * shape, int n_dims){
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
};

template <typename T>
class Op{
    public:
        Op(Tensor<T> * output, int n_in, ...);
        virtual ~Op();
        virtual void back() = 0;

    protected:
        int n_in;
        TensorHistory<T> ** inputs;
        TensorHistory<T> * output;
};
template <typename T>
Op<T>::Op(Tensor<T> * output, int n_in, ...) {
    //Read in the inputs
    Tensor<T>* in[n_in];
    va_list ins;
    va_start(ins, n_in);
    for(int i=0; i<n_in; i++) in[i] = va_arg(ins, Tensor<T>*);
    //Initialize all memory
    this->n_in = n_in;
    //Create output history
    this->inputs = new TensorHistory<T>* [n_in];
    this->output = new TensorHistory<T>(output, output->getDims() , output->getNDims());
    bool track = true;
    for(int i=0; i<n_in; i++) track &= in[i]->history();
    if(track) output->init_grad();
    for(int i=0; i<n_in; i++) {
        //Add to graph and init Grad
        if(track){
            this->inputs[i] = new TensorHistory<T>(in[i], in[i]->getDims(), in[i]->getNDims());
            in[i]->init_grad();
            in[i]->addChild(output);
            output->addParent(in[i]);
        }
    }
}
template <typename T>
Op<T>::~Op() {
    std::cout << "OP DELETE CALLED" << std::endl;
    for(int i=0; i<n_in; i++){
        delete inputs[i];
    }
    delete [] inputs;
    delete  output;
}
/********************************************************************************************/
/*                                          INDIVIAUAL OPS                                  */
/********************************************************************************************/
template <typename T>
class _ADD: public Op<T>{
    public:
    _ADD(Tensor<T>*output, Tensor<T>* input1, Tensor<T>* input2): Op<T>(output, 2, input1, input2) {}

    void back(){
        //How to add Must be defined without
        //Actually using the add operation
        Tensor<T> * err_sig  = this->output->tensor->getGrad();

        int arr[this->inputs[0]->n_dims];
        for(int i=0; i<this->inputs[0]->n_dims; i++) arr[i] = 0;

        for(int i=0; i < this->n_in; i++){
            int * shape = this->inputs[i]->shape;
            int n_dims = this->inputs[i]->n_dims;
            this->inputs[i]->tensor->reshape_grad(n_dims, shape);
            //Add the gradients
            iterator<T> it1(this->inputs[i]->tensor->getGrad() , arr);
            iterator<T> it2(err_sig, arr);
            for(int i=0; i<this->inputs[0]->tensor->getTotalElements(); i++){
                it1.next()+=it2.next();
            }
        }
    }
};

namespace OPS{
    /*
    Tensors created by the ops must manually be deleted
    Coming soon Tensor::deleteAll();
    */
    template <typename T>
    Tensor<T> * ADD(Tensor<T>* input1, Tensor<T> * input2) {
        //Until broadcasting is implemented
        //We also must garentee the tensors have the same shape
        assert(input1->getTotalElements() == input2->getTotalElements());

        T  new_data[input1->getTotalElements()];
        int arr[input1->getNDims()];
        for(int i=0; i<input1->getNDims(); i++) arr[i] = 0;
        iterator<T> it1(input1, arr);
        iterator<T> it2(input2, arr);

        for(int i=0; i<input1->getTotalElements(); i++){
            new_data[i] = it1.next() + it2.next();
        }
        Tensor<T> * out = new Tensor<T>(new_data, input1->getNDims(), input1->getDims());
        _ADD<T> * add = new _ADD<T>(out, input1, input2);
        out->setOP(dynamic_cast<Op<T>*>(add));
        return out;
    }
}


#endif