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
    this->inputs = new TensorHistory<T>* [n_in];
    this->output = new TensorHistory<T>(output, output->getDims() , output->getNDims());
    bool track = true;
    for(int i=0; i<n_in; i++) track &= in[i]->history();
    if(track) output->init_grad();

    //Initialize input tensor history 
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
        assert(this->inputs[0]->tensor->history());
        assert(this->inputs[1]->tensor->history());

        //Get dL/dout
        Tensor<T> * err_sig  = this->output->tensor->getGrad();

        //Initialize and Set starting index
        int arr[this->inputs[0]->n_dims];
        for(int i=0; i<this->inputs[0]->n_dims; i++) arr[i] = 0;

        //Compute Gradients for each input (2)
        for(int i=0; i < this->n_in; i++){
            int * shape = this->inputs[i]->shape;
            int n_dims = this->inputs[i]->n_dims;

            //Shape the gradient to the historical state so shapes
            //Match correctly
            this->inputs[i]->tensor->reshape_grad(n_dims, shape);
            //Add the gradients
            iterator<T> it1(this->inputs[i]->tensor->getGrad() , NULL,  arr);
            iterator<T> it2(err_sig, NULL,  arr);
            for(int i=0; i<this->inputs[0]->tensor->getTotalElements(); i++){
                //dL/din = dL/dout * 1
                it1.next()+=it2.next();
            }
        }
    }
};

template <typename T>
class _MULT: public Op<T> {
    public:
    _MULT(Tensor<T>*output, Tensor<T>* input1, Tensor<T>* input2): Op<T>(output, 2, input1, input2) {}

    void back(){
        assert(this->inputs[0]->tensor->history());
        assert(this->inputs[1]->tensor->history());

        //Get dL/dout
        Tensor<T> * err_sig  = this->output->tensor->getGrad();
        

        /*Save original shape:
        *We must save the shape because in order to compute
        *The gradient the tensor must be reshaped to its historical
        *Shape */

        int in1_dims[this->inputs[0]->tensor->getNDims()];
        int in1_dimc = this->inputs[0]->tensor->getNDims();
        //Copy the values over so when we call reshape they dont
        //Change because we only have a pointer to the shape
        for(int i=0; i<in1_dimc; i++) 
            in1_dims[i] = this->inputs[0]->tensor->getDims()[i];

        int in2_dims[this->inputs[1]->tensor->getNDims()];
        int in2_dimc = this->inputs[1]->tensor->getNDims();
        //Copy the values over so when we call reshape they dont
        //Change because we only have a pointer to the shape
        for(int i=0; i<in2_dimc; i++) 
            in2_dims[i] = this->inputs[1]->tensor->getDims()[i];

        //Reshape to the historical shape
        //Note: reshape checks the current shape
        //      if they match it will just return
        this->inputs[0]->tensor->reshape(this->inputs[0]->n_dims, this->inputs[0]->shape);
        this->inputs[1]->tensor->reshape(this->inputs[1]->n_dims, this->inputs[1]->shape);

        //Init and Set index to 0
        int arr[this->inputs[0]->n_dims];
        for(int i=0; i<this->inputs[0]->n_dims; i++) arr[i] = 0;

        for(int i=0; i < this->n_in; i++){
            //Get the historical shape to reshape the gradient
            int * shape = this->inputs[i]->shape;
            int n_dims = this->inputs[i]->n_dims;
            this->inputs[i]->tensor->reshape_grad(n_dims, shape);

            //Set up iterators
            iterator<T> it1(this->inputs[i]->tensor->getGrad() , NULL,  arr);
            iterator<T> it2(err_sig, NULL,  arr);
            for(int j=0; j<this->inputs[0]->tensor->getTotalElements(); j++){
                //Get the other tensor 
                //Example: Y = WX (scalars), dY/dX = W, dY/dW = X
                Tensor<T> * other_tensor = this->inputs[(i+1)%2]->tensor;

                //Must be paired with a delete

                it1.getCurr(arr);//Fix when implementing broadcasting

                //Get the value to multiply by
                T mult = other_tensor->get(arr);
                it1.next() += (it2.next() * mult);

            }
        }
        //Reshape back to original shape
        this->inputs[0]->tensor->reshape(in1_dimc, in1_dims);
        this->inputs[1]->tensor->reshape(in2_dimc, in2_dims);
    }
};

template <typename T>
class _MatMul: public Op<T>{
    public:
    _MatMul(Tensor<T>*output, Tensor<T>* input1, Tensor<T>* input2): Op<T>(output, 2, input1, input2) {}

    void back(){}
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
        assert(input1->getNDims() == input2->getNDims());
        for(int i=0; i<0; i++) 
            assert(input1->getDims()[i] == input2->getDims()[i]);

        T  new_data[input1->getTotalElements()];
        int arr[input1->getNDims()];
        for(int i=0; i<input1->getNDims(); i++) arr[i] = 0;
        iterator<T> it1(input1, NULL,  arr);
        iterator<T> it2(input2, NULL,  arr);

        for(int i=0; i<input1->getTotalElements(); i++){
            new_data[i] = it1.next() + it2.next();
        }
        Tensor<T> * out = new Tensor<T>(new_data, input1->getNDims(), input1->getDims());
        _ADD<T> * add = new _ADD<T>(out, input1, input2);
        out->setOP(dynamic_cast<Op<T>*>(add));
        return out;
    }

    template <typename T>
    Tensor<T> * MULT(Tensor<T>* input1, Tensor<T> * input2) {
        //Until broadcasting is implemented
        //We also must garentee the tensors have the same shape
        assert(input1->getNDims() == input2->getNDims());
        for(int i=0; i<0; i++) 
            assert(input1->getDims()[i] == input2->getDims()[i]);

        //Allocate new Tensors data
        T  new_data[input1->getTotalElements()];

        //Set up iterators
        int arr[input1->getNDims()];
        for(int i=0; i<input1->getNDims(); i++) arr[i] = 0;
        iterator<T> it1(input1, NULL,  arr);
        iterator<T> it2(input2, NULL, arr);

        //Multiply element wise
        for(int i=0; i<input1->getTotalElements(); i++){
            new_data[i] = it1.next() * it2.next();
        }
        Tensor<T> * out = new Tensor<T>(new_data, input1->getNDims(), input1->getDims());
        _MULT<T> * add = new _MULT<T>(out, input1, input2);
        out->setOP(dynamic_cast<Op<T>*>(add));
        return out;
    }
    //OPS HELPERS
    template <typename T>
    T _dot(Tensor<T>* input1, Tensor<T> * input2, int * index1, int * index2) {
        assert(input1->getDims()[input1->getNDims()-1] == input2->getDims()[input2->getNDims()-1]);


        iterator<T> it1(input1, NULL, index1);
        iterator<T> it2(input2, NULL, index2);
        T accum = 0;

        for(int i=0; i<input1->getDims()[input1->getNDims()-1]; i++){
            accum += (it1.next() * it2.next());
        }
        return accum;
    }
    template <typename T>
    Tensor<T> * _matmul(Tensor<T> * input1, Tensor<T> * input2) {
        //Shapes must match except for the last 2 layers
        assert(input1->getDims()[input1->getNDims()-1] == input2->getDims()[input2->getNDims()-2]);
        assert(input1->getNDims() == input2->getNDims());

        //Initialize out tensor
        int arr[input1->getNDims()];
        for(int i=0; i<input1->getNDims()-1; i++) arr[i] = input1->getDims()[i];
        arr[input1->getNDims()-1] = input2->getDims()[input2->getNDims()-1];
        Tensor<T> * out = new Tensor<T>(input1->getNDims(), arr);
        out->no_history();

        //set up input2 for the dot product
        input2->no_history();
        input2->transpose();
        bool was_cont = input2->is_contiguous();

        //set up iterators 1 & 2
        int order[input1->getNDims()];
        for(int i=0; i<input1->getNDims(); i++) {
            order[i] = i+1;
            arr[i] = 0;
        }
        order[input1->getNDims()-1] = 0;

        iterator<T> it1(input1, order, arr);
        iterator<T> it2(input2, order, arr);

        //set up iterator 3
        for(int i=0; i<input1->getNDims(); i++) order[i] = i + 2;
        order[input1->getNDims()-1] = 1;
        order[input1->getNDims()-2] = 0;
        iterator<T> it3(input1, order, arr);

        int divisor = input2->getDims()[input2->getNDims()-2]  * input2->getDims()[input2->getNDims()-1];   
        int loop_bound =  input2->getTotalElements() / divisor;
        int loop_bound1 = input2->getDims()[input2->getNDims()-2];
        int loop_bound2 = input1->getDims()[input1->getNDims()-2];

        //Multiply Tensors
        for(int _=0; _<loop_bound; _++){
            for(int i=0; i<loop_bound1; i++){
                //Store it2 current index in arr
                it2.getCurr(arr);
                for(int j=0; j<loop_bound2; j++){

                    it1.getCurr(order);
                    T element  = _dot(input1, input2, order, arr);

                    order[input1->getNDims()-1] = arr[input1->getNDims()-2];

                    out->get(order) = element;
                    it1.next();
                }
                it2.next();
                it3.getCurr(order);
                it1.setCurr(order);
            }
            it3.next();
            it3.getCurr(order);
            it1.setCurr(order);
        }
        //set input2 back to its orignal state
        input2->transpose();
        input2->use_history();
        input2->set_contiguous(was_cont);

        return out;
    }
}


#endif