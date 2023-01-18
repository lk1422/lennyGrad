#ifndef OPS_H_
#define OPS_H_

#include <algorithm>
#include <iostream>
#include <cmath>
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

    template <typename T>
    void inplace_mult(Tensor<T> * input1, Tensor<T> * input2);

    template <typename T>
    void inplace_mult_recip(Tensor<T> * input1, Tensor<T> * input2);

    template <typename T>
    void inplace_sub(Tensor<T> * input1, Tensor<T> * input2);
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

template <typename T>
class _SUB: public Op<T>{
    public:
    _SUB(Tensor<T>*output, Tensor<T>* input1, Tensor<T>* input2): Op<T>(output, 2, input1, input2) {}

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
        }
        //Add the gradients
        OPS::inplace_add(this->inputs[0]->getGrad(), err_sig);
        //Sub the gradients
        OPS::inplace_sub(this->inputs[1]->getGrad(), err_sig);
    }
};


//Operator Descendent for element wise product
template <typename T>
class _MULT: public Op<T> {
    public:
    _MULT(Tensor<T>*output, Tensor<T>* input1, Tensor<T>* input2): Op<T>(output, 2, input1, input2) {}

    void back(){
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
        }

        
        Tensor<T> err_sig2(*err_sig);
        OPS::inplace_mult(err_sig, this->inputs[1]);
        OPS::inplace_add(this->inputs[0]->getGrad(), err_sig);

        OPS::inplace_mult(&err_sig2, this->inputs[0]);
        OPS::inplace_add(this->inputs[1]->getGrad(), &err_sig2);

    }
};

template <typename T>
class _DIV: public Op<T> {
    public:
    _DIV(Tensor<T>*output, Tensor<T>* input1, Tensor<T>* input2): Op<T>(output, 2, input1, input2) {}

    void back(){
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
        }

        Tensor<T> err_sig2(*err_sig);
        OPS::inplace_mult_recip(err_sig, this->inputs[1]);
        OPS::inplace_add(this->inputs[0]->getGrad(), err_sig);

        iterator it1 = err_sig2.begin();
        iterator it2 = this->inputs[1]->getGrad()->begin();
        iterator it3 = this->inputs[0]->begin();
        iterator it4 = this->inputs[1]->begin();
        for(int i=0; i<err_sig2.getTotalElements(); i++){
            T denom = it4.next();
            it2.next() += (-it1.next()) * (it3.next()/(denom * denom));
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

template <typename T>
class _NEG: public Op<T>{
    public:
    _NEG(Tensor<T>*output, Tensor<T>* input): Op<T>(output, 1, input) {}

    void back(){
        assert(this->inputs[0]->history());

        //Get dL/dout
        Tensor<T> * err_sig  = this->output->getGrad();

        //Compute Gradients for each input (2)
        const int * shape = this->inputs[0]->getDims();
        const int n_dims = this->inputs[0]->getNDims();

        //Shape the gradient to the historical state so shapes
        //Match correctly
        this->inputs[0]->reshape_grad(n_dims, shape);

        //Add the gradients
        OPS::inplace_sub(this->inputs[0]->getGrad(), err_sig);
    }
};

template <typename T>
class _ReLU: public Op<T>{
    public:
    _ReLU(Tensor<T>*output, Tensor<T>* input): Op<T>(output, 1, input) {}

    void back(){
        assert(this->inputs[0]->history());

        //Get dL/dout
        Tensor<T> * err_sig  = this->output->getGrad();

        //Compute Gradients for each input (2)
        const int * shape = this->inputs[0]->getDims();
        const int n_dims = this->inputs[0]->getNDims();

        //Shape the gradient to the historical state so shapes
        //Match correctly
        this->inputs[0]->reshape_grad(n_dims, shape);

        iterator it1 = err_sig->begin();
        iterator it2 = this->inputs[0]->begin();
        iterator it3 = this->inputs[0]->getGrad()->begin();

        for(int i=0; i<this->output->getTotalElements(); i++){
            T next = it2.next();
            it3.next() += (std::max(next, (T)0)/next) * it1.next();
        }
    }
};

template <typename T>
class _EXP: public Op<T>{
    public:
    _EXP(Tensor<T>*output, Tensor<T>* input): Op<T>(output, 1, input) {}

    void back(){
        assert(this->inputs[0]->history());

        //Get dL/dout
        Tensor<T> * err_sig  = this->output->getGrad();

        //Compute Gradients for each input (2)
        const int * shape = this->inputs[0]->getDims();
        const int n_dims = this->inputs[0]->getNDims();

        //Shape the gradient to the historical state so shapes
        //Match correctly
        this->inputs[0]->reshape_grad(n_dims, shape);

        OPS::inplace_mult(err_sig, this->output);
        OPS::inplace_add(this->inputs[0]->getGrad(), err_sig);
    }
};

template <typename T>
class _PAD: public Op<T>{
    public:
    _PAD(Tensor<T>*output, Tensor<T>* input): Op<T>(output, 1, input) {}

    void back(){
        assert(this->inputs[0]->history());

        //Get dL/dout
        Tensor<T> * err_sig  = this->output->getGrad();

        //Compute Gradients for each input (2)
        const int * shape = this->inputs[0]->getDims();
        const int n_dims = this->inputs[0]->getNDims();

        //Shape the gradient to the historical state so shapes
        //Match correctly
        this->inputs[0]->reshape_grad(n_dims, shape);

        //compute padx, pady
        int pady = (this->output->getDims()[this->output->getNDims()-1] - this->inputs[0]->getDims()[this->inputs[0]->getNDims()-1] )/2;
        int padx = (this->output->getDims()[this->output->getNDims()-2] - this->inputs[0]->getDims()[this->inputs[0]->getNDims()-2] )/2;

        //Copy gradients over
        iterator it = this->inputs[0]->getGrad()->begin();
        int index[this->inputs[0]->getNDims()];
        for(int i=0; i<this->inputs[0]->getTotalElements(); i++){
            it.getCurr(index);
            index[this->inputs[0]->getNDims()-2] += padx;
            index[this->inputs[0]->getNDims()-1] += pady;
            it.next() += err_sig->get(index);
        }
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

        int test[input1->getNDims()];
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
                    it1.getCurr(test);
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
    void inplace_sub(Tensor<T> * input1, Tensor<T> * input2) {
        //ADD INPLACE INTO input1
        //assert same shape
        assert(input1->getNDims() == input2->getNDims());
        for(int i=0; i<input1->getNDims(); i++)
            assert(input1->getDims()[i] == input2->getDims()[i]);

        iterator<T> it1 = input1->begin();
        iterator<T> it2 = input2->begin();
        //sub element wise
        for(int i=0; i<input1->getTotalElements(); i++){
            it1.next() -= it2.next();
        }
    }

    template <typename T>
    void inplace_add(Tensor<T> * input1, Tensor<T> * input2) {
        //ADD INPLACE INTO input1
        //assert same shape
        assert(input1->getNDims() == input2->getNDims());
        for(int i=0; i<input1->getNDims(); i++)
            assert(input1->getDims()[i] == input2->getDims()[i]);

        iterator<T> it1 = input1->begin();
        iterator<T> it2 = input2->begin();
        //add element wise
        for(int i=0; i<input1->getTotalElements(); i++){
            it1.next() += it2.next();
        }
    }

    template <typename T>
    void inplace_mult_recip(Tensor<T> * input1, Tensor<T> * input2) {
        //ADD INPLACE INTO input1
        //assert same shape
        assert(input1->getNDims() == input2->getNDims());
        for(int i=0; i<input1->getNDims(); i++)
            assert(input1->getDims()[i] == input2->getDims()[i]);

        iterator<T> it1 = input1->begin();
        iterator<T> it2 = input2->begin();
        //mult element wise
        for(int i=0; i<input1->getTotalElements(); i++){
            it1.next() *= (1/it2.next());
        }
    }

    template <typename T>
    void inplace_mult(Tensor<T> * input1, Tensor<T> * input2) {
        //ADD INPLACE INTO input1
        //assert same shape
        assert(input1->getNDims() == input2->getNDims());
        for(int i=0; i<input1->getNDims(); i++)
            assert(input1->getDims()[i] == input2->getDims()[i]);

        iterator<T> it1 = input1->begin();
        iterator<T> it2 = input2->begin();
        //mult element wise
        for(int i=0; i<input1->getTotalElements(); i++){
            it1.next() *= it2.next();
        }
    }



/***************************
*     BINARY OPERATIONS    * 
****************************/

    template <typename T>
    Tensor<T> * ADD(Tensor<T>* input1, Tensor<T> * input2) {
        assert(input1->getTotalElements() == input2->getTotalElements());
        assert(input1->getNDims() == input2->getNDims());
        for(int i=0; i<input1->getNDims(); i++) 
            assert(input1->getDims()[i] == input2->getDims()[i]);

        //Set Up iterators
        iterator<T> it1 = input1->begin();
        iterator<T> it2 = input2->begin();

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
    Tensor<T> * SUB(Tensor<T>* input1, Tensor<T> * input2) {
        assert(input1->getTotalElements() == input2->getTotalElements());
        assert(input1->getNDims() == input2->getNDims());
        for(int i=0; i<input1->getNDims(); i++) 
            assert(input1->getDims()[i] == input2->getDims()[i]);

        //Set Up iterators
        iterator<T> it1 = input1->begin();
        iterator<T> it2 = input2->begin();

        //Add data
        T  new_data[input1->getTotalElements()];
        for(int i=0; i<input1->getTotalElements(); i++){
            new_data[i] = it1.next() - it2.next();
        }
        //Create out tensor
        Tensor<T> * out = new Tensor<T>(new_data, input1->getNDims(), input1->getDims());
        _SUB<T> * add = new _SUB<T>(out, input1, input2);
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
        iterator<T> it1 = input1->begin();
        iterator<T> it2 = input2->begin();

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
    Tensor<T> * DIV(Tensor<T>* input1, Tensor<T> * input2) {
        assert(input1->getTotalElements() == input2->getTotalElements());
        assert(input1->getNDims() == input2->getNDims());
        for(int i=0; i<input1->getNDims(); i++) 
            assert(input1->getDims()[i] == input2->getDims()[i]);

        //Set Up iterators
        iterator<T> it1 = input1->begin();
        iterator<T> it2 = input2->begin();

        //Add data
        T  new_data[input1->getTotalElements()];
        for(int i=0; i<input1->getTotalElements(); i++){
            new_data[i] = it1.next() / it2.next();
        }

        //Create out tensor
        Tensor<T> * out = new Tensor<T>(new_data, input1->getNDims(), input1->getDims());
        _DIV<T> * div = new _DIV<T>(out, input1, input2);
        out->setOP(dynamic_cast<Op<T>*>(div));
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

/***************************
*     UNARY OPERATIONS     * 
****************************/
    template <typename T>
    Tensor<T> * NEG(Tensor<T>* input) {

        iterator<T> it = input->begin();
        //Add data
        T  new_data[input->getTotalElements()];
        for(int i=0; i<input->getTotalElements(); i++){
            new_data[i] = -it.next();
        }
        Tensor<T> * out = new Tensor<T>(new_data, input->getNDims(), input->getDims());

        // Set up Out Tensor
        _NEG<T> * neg = new _NEG<T>(out, input);
        out->setOP(dynamic_cast<Op<T>*>(neg));
        return out;
    }
    template <typename T>
    Tensor<T> * ReLU(Tensor<T>* input) {

        iterator<T> it = input->begin();

        //Add data
        T  new_data[input->getTotalElements()];
        for(int i=0; i<input->getTotalElements(); i++){
            new_data[i] = std::max(0.0, (double)it.next());
        }
        Tensor<T> * out = new Tensor<T>(new_data, input->getNDims(), input->getDims());

        // Set up Out Tensor
        _ReLU<T> * rel = new _ReLU<T>(out, input);
        out->setOP(dynamic_cast<Op<T>*>(rel));
        return out;
    }

    template <typename T>
    Tensor<T> * EXP(Tensor<T>* input) {

        iterator<T> it = input->begin();

        //Add data
        T  new_data[input->getTotalElements()];
        for(int i=0; i<input->getTotalElements(); i++){
            new_data[i]  = std::exp(it.next());
        }
        Tensor<T> * out = new Tensor<T>(new_data, input->getNDims(), input->getDims());

        // Set up Out Tensor
        _EXP<T> * exp = new _EXP<T>(out, input);
        out->setOP(dynamic_cast<Op<T>*>(exp));
        return out;
    }

    template <typename T>
    Tensor<T> * PAD(Tensor<T>* input, int padx=2, int pady=2, int pad_val = 0) {
        assert(input->getNDims() >= 2);

        std::pair<int,int> pad = std::make_pair(padx, pady);

        //Get new shape
        int dims[input->getNDims()];
        copyElements(input->getNDims(), dims, input->getDims());
        dims[input->getNDims()-2] += 2*pad.first;
        dims[input->getNDims()-1] += 2*pad.second;
        //Create out tensor
        Tensor<T> * out = new Tensor<T>(input->getNDims(), dims);
        out->setAll(pad_val);//0 pad
        //Copy elements over
        iterator<T> it = input->begin();

        int index[input->getNDims()];
        for(int i=0; i<input->getTotalElements(); i++){
            it.getCurr(index);
            index[input->getNDims()-2] += pad.first;
            index[input->getNDims()-1] += pad.second;
            out->get(index) = it.next();
        }

        // Set up Out Tensor
        _PAD<T> * pad_ = new _PAD<T>(out, input);
        out->setOP(dynamic_cast<Op<T>*>(pad_));
        return out;
    }
}


template <typename T>
T convAt(Tensor<T> * X, Tensor<T> * K, std::pair<int,int> s, const int * AT){
    assert(X->getNDims() == 4);
    assert(K->getNDims() == 4);
    //Allocate indexes

    T accum = 0;
    for(int l=0; l<X->getDims()[1]; l++){
        for(int m=0; m<K->getDims()[2]; m++){
            for(int n=0; n<K->getDims()[3]; n++){
                int k_ind[] = {AT[1], l, m, n};
                int x_ind[] = {AT[0], l, s.first*AT[2] +m, s.second*AT[3] + n};
                accum += X->get(x_ind) * K->get(k_ind);
            }
        }
    }
    return accum;
}

template <typename T>
Tensor<T> * conv(Tensor<T> * X, Tensor<T> * K, std::pair<int,int> s){
    assert(X->getNDims() == 4);
    assert(K->getNDims() == 4);
    //Calculate out shape
    int height = (X->getDims()[2] - K->getDims()[2] + 1)/s.first;
    int width = (X->getDims()[3] - K->getDims()[3] + 1)/s.second;
    Tensor<T>* out = new Tensor<T>(4, X->getDims()[0], K->getDims()[0], height, width);
    for(int b=0; b<out->getDims()[0]; b++){
        for(int i=0; i<out->getDims()[1]; i++){
            for(int j=0; j<out->getDims()[2]; j++){
                for(int k=0; k<out->getDims()[3]; k++){
                    int index[] = {b, i, j, k};
                    out->get(index) = convAt(X, K, s, index);
                }
            }
        }
    }
    return out;
}


#endif