#ifndef OPS_H_
#define OPS_H_

#include <iostream>
#include "tensor.h"
#include "utils.h"

/*
    FORWARD DECLARATIONS
*/
namespace OPS{
    template <typename T>
    Tensor<T> * _matmul(Tensor<T> * input1, Tensor<T> * input2);

    template <typename T>
    void inplace_add(Tensor<T> * input1, Tensor<T> * input2);
}

template <typename T>
class Op{
/*
    Abstract Class for operations
    All descendents of the Op class
    Must implement the back() method
*/
    public:
        Op(Tensor<T> * output, int n_in, ...);
        virtual ~Op() { delete [] inputs; }
        virtual void back() = 0;

    protected:
        int n_in;
        Tensor<T> ** inputs;
        Tensor<T> * output;
        bool history;
};

/*Op Class Constructor*/
template <typename T>
Op<T>::Op(Tensor<T> * output, int n_in, ...) {

    //Read in the inputs
    Tensor<T>* in[n_in];
    va_list ins;
    va_start(ins, n_in);
    for(int i=0; i<n_in; i++) in[i] = va_arg(ins, Tensor<T>*);

    //Initialize all memory
    this->n_in = n_in;
    this->inputs = new Tensor<T>* [n_in];
    this->output = output;
    bool track = true;
    for(int i=0; i<n_in; i++) track &= in[i]->history();
    if(track) output->init_grad();
    this->history = track;

    //Initialize input tensor history 
    for(int i=0; i<n_in; i++) {
        //Add to graph and init Grad
        if(track){
            this->inputs[i] = in[i];
            in[i]->init_grad();
            in[i]->addChild(output);
            output->addParent(in[i]);
        }
    }
}

/********************************************************************************************/
/*                                          INDIVIAUAL OPS                                  */
/********************************************************************************************/

//Operator Descendent for addition
template <typename T>
class _ADD: public Op<T>{
    public:
    _ADD(Tensor<T>*output, Tensor<T>* input1, Tensor<T>* input2): Op<T>(output, 2, input1, input2) {}

    void back(){
        assert(this->inputs[0]->history());
        assert(this->inputs[1]->history());

        //Get dL/dout
        Tensor<T> * err_sig  = this->output->getGrad();

        //Compute Gradients for each input (2)
        for(int i=0; i < this->n_in; i++){
            const int * shape = this->inputs[i]->getDims();
            const int n_dims = this->inputs[i]->getNDims();

            //Shape the gradient to the historical state so shapes
            //Match correctly
            this->inputs[i]->reshape_grad(n_dims, shape);

            //Add the gradients
            OPS::inplace_add(this->inputs[i]->getGrad(), err_sig);
        }
    }
};

//Operator Descendent for element wise product
template <typename T>
class _MULT: public Op<T> {
    public:
    _MULT(Tensor<T>*output, Tensor<T>* input1, Tensor<T>* input2): Op<T>(output, 2, input1, input2) {}

    void back(){
        assert(this->inputs[0]->getTotalElements() == this->inputs[1]->getTotalElements());
        assert(this->inputs[0]->history());
        assert(this->inputs[0]->is_grad_init());
        assert(this->inputs[1]->is_grad_init());
        assert(this->inputs[1]->history());
        assert(this->inputs[1]->history());

        //Get dL/dout
        Tensor<T> * err_sig  = this->output->getGrad();
        
        //Init and Set index to 0
        int arr[this->inputs[0]->getNDims()];

        int zeros[this->inputs[0]->getNDims()];
        setAllElements(this->inputs[0]->getNDims(), zeros, 0);

        for(int i=0; i < this->n_in; i++){
            //Get the historical shape to reshape the gradient
            const int * shape = this->inputs[i]->getDims();
            const int n_dims = this->inputs[i]->getNDims();
            this->inputs[i]->reshape_grad(n_dims, shape);

            //Set up iterators
            iterator<T> it1(this->inputs[i]->getGrad() , NULL,  zeros);
            iterator<T> it2(err_sig, NULL,  zeros);
            for(int j=0; j<this->inputs[i]->getTotalElements(); j++){
                //Get the other tensor 
                //Example: Y = WX (scalars), dY/dX = W, dY/dW = X
                Tensor<T> * other_tensor = this->inputs[(i+1)%2];

                it1.getCurr(arr);//Fix when implementing broadcasting

                //Get the value to multiply by
                T mult = other_tensor->get(arr);
                it1.next() += it2.next() * mult;
            }
        }
    }
};

//Operator Descendent for tensor product
template <typename T>
class _MatMul: public Op<T>{
    public:
    _MatMul(Tensor<T>*output, Tensor<T>* input1, Tensor<T>* input2): Op<T>(output, 2, input1, input2) {}

    void back(){
        Tensor<T> * err_sig  = this->output->getGrad();

        //Get the historical shape to reshape the gradient
        for(int i=0; i < this->n_in; i++){
            const int * shape = this->inputs[i]->getDims();
            const int n_dims = this->inputs[i]->getNDims();
            this->inputs[i]->reshape_grad(n_dims, shape);
        }

        //Compute Gradient dL/dX
        this->inputs[1]->transpose();
        bool was_cont = this->inputs[1]->is_contiguous();

        Tensor<T> * dx = OPS::_matmul(err_sig, this->inputs[1]);
        OPS::inplace_add(this->inputs[0]->getGrad(), dx);

        this->inputs[1]->transpose();
        this->inputs[1]->set_contiguous(was_cont);

        //Compute Gradient dL/dW
        this->inputs[0]->transpose();
        was_cont = this->inputs[0]->is_contiguous();

        Tensor<T> * dw = OPS::_matmul(this->inputs[0], err_sig);
        OPS::inplace_add(this->inputs[1]->getGrad(), dw);

        this->inputs[0]->transpose();
        this->inputs[0]->set_contiguous(was_cont);

        delete dx;
        delete dw;
    }
};

namespace OPS{

/*
    OPS Helper Functions
*/
    template <typename T>
    T _dot(Tensor<T>* input1, Tensor<T> * input2, int * index1, int * index2) {
        assert(input1->getDims()[input1->getNDims()-1] == input2->getDims()[input2->getNDims()-1]);


        //Set up iterators and initialize accumulator
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
        copyElements(input1->getNDims(), arr, input1->getDims());
        arr[input1->getNDims()-1] = input2->getDims()[input2->getNDims()-1];  //Correct last dimension
        Tensor<T> * out = new Tensor<T>(input1->getNDims(), arr);
        out->no_history();

        //set up input2 for the dot product
        input2->transpose();
        bool was_cont = input2->is_contiguous();

        //set up iterators 1 & 2
        setAllElements(input1->getNDims(), arr, 0);

        int order[input1->getNDims()];
        for(int i=0; i<input1->getNDims(); i++) order[i] = i+1;
        order[input1->getNDims()-1] = 0;

        iterator<T> it1(input1, order, arr);
        iterator<T> it2(input2, order, arr);

        //set up iterator 3
        for(int i=0; i<input1->getNDims(); i++) order[i] = i + 2;
        order[input1->getNDims()-1] = 1;
        order[input1->getNDims()-2] = 0;
        iterator<T> it3(input1, order, arr);

        //Set Loop Bounds
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
        input2->set_contiguous(was_cont);

        return out;
    }

    template <typename T>
    void inplace_add(Tensor<T> * input1, Tensor<T> * input2) {
        //ADD INPLACE INTO input1
        //assert same shape
        assert(input1->getNDims() == input2->getNDims());
        for(int i=0; i<input1->getNDims(); i++)
            assert(input1->getDims()[i] == input2->getDims()[i]);

        int arr[input1->getNDims()];
        for(int i=0; i<input1->getNDims(); i++) arr[i] = 0;
        iterator<T> it1(input1, NULL,  arr);
        iterator<T> it2(input2, NULL, arr);
        //add element wise
        for(int i=0; i<input1->getTotalElements(); i++){
            it1.next() += it2.next();
        }
    }

/***************************
*       OPERATIONS         * 
****************************/

    template <typename T>
    Tensor<T> * ADD(Tensor<T>* input1, Tensor<T> * input2) {
        assert(input1->getTotalElements() == input2->getTotalElements());
        assert(input1->getNDims() == input2->getNDims());
        for(int i=0; i<input1->getNDims(); i++) 
            assert(input1->getDims()[i] == input2->getDims()[i]);

        //Set Up iterators
        int arr[input1->getNDims()];
        setAllElements(input1->getNDims(), arr, 0);
        iterator<T> it1(input1, NULL,  arr);
        iterator<T> it2(input2, NULL,  arr);

        //Add data
        T  new_data[input1->getTotalElements()];
        for(int i=0; i<input1->getTotalElements(); i++){
            new_data[i] = it1.next() + it2.next();
        }
        //Create out tensor
        Tensor<T> * out = new Tensor<T>(new_data, input1->getNDims(), input1->getDims());
        _ADD<T> * add = new _ADD<T>(out, input1, input2);
        out->setOP(dynamic_cast<Op<T>*>(add));
        return out;
    }

    template <typename T>
    Tensor<T> * MULT(Tensor<T>* input1, Tensor<T> * input2) {
        assert(input1->getTotalElements() == input2->getTotalElements());
        assert(input1->getNDims() == input2->getNDims());
        for(int i=0; i<input1->getNDims(); i++) 
            assert(input1->getDims()[i] == input2->getDims()[i]);

        //Set Up iterators
        int arr[input1->getNDims()];
        setAllElements(input1->getNDims(), arr, 0);
        iterator<T> it1(input1, NULL,  arr);
        iterator<T> it2(input2, NULL,  arr);

        //Add data
        T  new_data[input1->getTotalElements()];
        for(int i=0; i<input1->getTotalElements(); i++){
            new_data[i] = it1.next() * it2.next();
        }

        //Create out tensor
        Tensor<T> * out = new Tensor<T>(new_data, input1->getNDims(), input1->getDims());
        _MULT<T> * add = new _MULT<T>(out, input1, input2);
        out->setOP(dynamic_cast<Op<T>*>(add));
        return out;
    }

    template <typename T>
    Tensor<T> * MatMul(Tensor<T>* input1, Tensor<T> * input2) {
        assert(input1->getNDims() == input2->getNDims());

        //Multiply Tensors
        Tensor<T> * out = _matmul(input1, input2);
        out->use_history();

        // Set up Out Tensor
        _MatMul<T> * matmul = new _MatMul<T>(out, input1, input2);
        out->setOP(dynamic_cast<Op<T>*>(matmul));
        return out;
    }
}
#endif