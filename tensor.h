#ifndef TENSOR_H_
#define TENSOR_H_

#include <iostream>
#include <vector>
#include <cstdarg>
#include <iomanip>
#include <cassert>
#include <utility>
#include <random>

#include "iterator.h"

template <typename T> class Op;

template <typename T>
class Tensor {
    public:
        Tensor(int n_dims, ...);
        Tensor(const T * data, int n_dims, const int * dims);
        Tensor(int n_dims, const int * dims);
        ~Tensor();

        Tensor(const Tensor<T>& tensor);
        Tensor<T> & operator=(const Tensor<T>& tensor);

        //Methods
        /***************************************************************
        * T& get(int dims, ...) const;
        * 
        * Parameters:
        *   -dims -> This variable should be set to the number
        *            of dims of the tensor, however it is just
        *            here to mount the variable arguments off of
        *            in other words it doesn't affect the output
        *            but must be provided.
        *
        *   - ... -> This is for the dimensional indices
        *            passing in less than the total amount of
        *            dimensions in the tensor leads to undefined
        *            behavior.
        *  Returns:
        *       The value stored at tensor[i, j, k, ... , z]
        ***************************************************************/
        T& get(int dims, ...) const;

        /***************************************************************
        * T& get(int * dims) const;
        * 
        * Parameters:
        *   -dims -> a pointer to an array of length
        *            n_dims, which stores the dimensional
        *            indices to be accessed
        *
        *  Returns:
        *       The value stored at tensor[i, j, k, ... , z]
        ***************************************************************/
        T& get(const int * dims) const;

        /***************************************************************
        * bool is_contiguous() const {return contiguous;}
        *  
        * Returns:
        *   A bool representing whether or not the internal array
        *   is contiguous
        ***************************************************************/
        bool is_contiguous() const {return contiguous;}

        /***************************************************************
        * std::pair<const int * , int> shape() const {return std::make_pair(dims, n_dims);}
        *
        *   Returns:
        *     A pair: first  -> a pointer to a const array storing the dimensions,
        *             second -> the total number of dimensions
        ***************************************************************/
        std::pair<const int * , int> shape() const {return std::make_pair(dims, n_dims);}

        /***************************************************************
        * std::vector<Tensor*> getChildren() const;
        *
        *   Returns:
        *       A vector storing the tensors children
        ***************************************************************/
        std::vector<Tensor*> getChildren() const { return children; }

        /***************************************************************
        * std::vector<Tensor*> getParents() const;
        *
        *   Returns:
        *       A vector storing the tensors parents
        ***************************************************************/
        std::vector<Tensor*> getParents() const { return parents; }

        /***************************************************************
        * const int * getMults() const { return mults; }
        *
        *   Returns:
        *       A pointer to the array storing the offsets for indexing the internal array
        *       The size of the array is n_dims (tensor.getNDims())
        ***************************************************************/
        const int * getMults() const { return mults; }

        /***************************************************************
        * const int * getDims() const { return dims; } 
        *
        *   Returns:
        *       A pointer to the array storing the tensors dimensions
        *       The size of the array is n_dims (tensor.getNDims())
        ***************************************************************/
        const int * getDims() const { return dims; } 

        /***************************************************************
        * const int * getLocalEls() const;
        *
        *   Returns:
        *       A pointer to the array storing the total amount
        *       of elements stored in that dimension of the data
        ***************************************************************/
        const int * getLocalEls() const { return local_els; }

        /***************************************************************
        * int getNDims() const { return n_dims; }
        *
        *   Returns:
        *       the total number of dimensions for the tensor
        ***************************************************************/
        int getNDims() const { return n_dims; }

        /***************************************************************
        * int getTotalElements() const { return n_els; }
        *
        *   Returns:
        *       the total number of elements of the array,
        *       this is also the length of the internal storage array
        *       for the tensor.
        ***************************************************************/
        int getTotalElements() const { return n_els; }

        /***************************************************************
        * int getIndex(int * dims) const;
        *
        *   Returns:
        *       The calculated internal array index for the element stored
        *       in the tensor at Tensor[i, j, k, ..., z]
        ***************************************************************/
        int getIndex(int * dims) const;

        /*Reshape Methods*/

        /***************************************************************
        * void as_contiguous();
        *
        *   Description:
        *       rearranges the internal storage array such that it is
        *       C contiguous (with respect to the rows).
        *       The internal contiguous array allows us to reshape the tensor
        *       into other dimensions. (Reshape calls this method before shaping the tensor)
        ***************************************************************/
        void as_contiguous();

        /***************************************************************
        * void set_contiguous(bool cont);
        *
        *   Description:
        *       sets the contiguous flag to the value of cont
        ***************************************************************/
        void set_contiguous(bool cont) { contiguous=cont; }

        /***************************************************************
        * void reshape(int dims, ...);
        *
        *   Description:
        *       shapes the tensor into the new dimension with n_dims now = to dims.
        *       Note the dims arg must be = to the amount of dimension arguements
        *       that follows or else the function will behave in a undefined manner

        *       NOTE
        *       the product of all dimensions also must be equal to the previous
        *       dimensions product (the total length of the internal array)
        ***************************************************************/
        void reshape(int dims, ...);

        /***************************************************************
        * void reshape(int n_dims, int * dims);
        *
        *   Description:
        *       shapes the tensor into the new dimension with n_dims now = to dims.
        *       Note the dims arg must be = to the length of the dims array
        *       that follows or else the function will behave in a undefined manner

        *       NOTE
        *       the product of all dimensions also must be equal to the previous
        *       dimensions product (the total length of the internal array)
        ***************************************************************/
        void reshape(int n_dims, const int * dims);

        /***************************************************************
        * void transpose();
        *
        *   Description:
        *       dimensionality of tensor must be >=2, Transpose swaps
        *       the last two dimensions of the tensor. This action
        *       makes the tensor no longer contiguous. (A double transpose
        *       even though preserving the contiguous property of the tensor
        *       will not be treated as if it is contiguous after ward).
        ***************************************************************/
        void transpose();

        /***************************************************************
        * void setAll(T val);
        *
        *   Description:
        *       sets all the elements to the value specified
        ***************************************************************/
        void setAll(T val){ for(int i=0; i<n_els; i++) data[i] = val;}

        /***************************************************************
        * void init_grad();
        *
        *   Description:
        *       Initializes the gradient to all 0s in the
        *       current shape of the tensor
        ***************************************************************/
        void init_grad();

        /***************************************************************
        * bool is_grad_init()
        *
        *   Returns:
        *       a bool represeting whether or not the grad
        *       of the tensor has been initialized
        ***************************************************************/
        bool is_grad_init() const { return grad_initialized; }

        /***************************************************************
        * void no_history()
        *
        *   Description:
        *       Sets the track_history() flag to false
        *       This means it will not be included in the
        *       Computational Graph
        ***************************************************************/
        void no_history() { track_history = false; }

        /***************************************************************
        * void use_history()
        *
        *   Description:
        *       Sets the use_history() flag to true
        *       This means it will be included in the
        *       Computational Graph
        ***************************************************************/
        void use_history() { track_history = true; }

        /***************************************************************
        * bool history()
        *
        *   Returns:
        *       Returns the boolean flag representing whether
        *       or not the tensors history should be tracked
        *       (If it should be added to the comp graph)
        ***************************************************************/
        bool history() const { return track_history; }


        /***************************************************************
        * void reshape_grad(int n_dims, int * dims);
        *
        *   Description:
        *       shapes the grad into the new dimension with n_dims now = to dims.
        *       Note the dims arg must be = to the length of the dims array
        *       that follows or else the function will behave in a undefined manner

        *       NOTE
        *       the product of all dimensions also must be equal to the previous
        *       dimensions product (the total length of the internal array)
        ***************************************************************/
        void reshape_grad(int n_dims,const int * dims);

        /***************************************************************
        * void setOP(Op * op);
        *
        *   Description:
        *       Sets the operation which was performed
        *       to create this tensor
        ***************************************************************/
        void setOP(Op<T> * op) {this->op=op;}

        /***************************************************************
        * void addChild(Tensor<T> * child);
        *
        *   Description:
        *       appends child tensor to child vector
        ***************************************************************/
        void addChild(Tensor<T> * child) { children.push_back(child); }

        /***************************************************************
        * void addParent(Tensor<T> * par);
        *
        *   Description:
        *       appends parent tensor to par vector
        ***************************************************************/
        void addParent(Tensor<T> * par) { parents.push_back(par); }

        /***************************************************************
        * void Tensor<T> * getGrad();
        *
        *   Description:
        *       Returns a pointer to the tensors gradient
        ***************************************************************/
        Tensor<T> * getGrad() const { assert(grad_initialized); return grad; }

        /***************************************************************
        * Op<T> * getOp() const;
        *
        *   Description:
        *       Returns a pointer to the operation which created
        *       this tensor
        ***************************************************************/
        Op<T> * getOp() const { return op; }

        /***************************************************************
        * T scalar() const;
        *
        *   Description:
        *       Returns the scalar value of the function
        *       IF it is a tensor with only one value
        *       ELSE an error is thrown
        ***************************************************************/
        T scalar() const {assert(n_els==1); return data[0];}

        /***************************************************************
        * void randn();
        *
        *   Description:
        *       Initializes the tensor to values in a normal
        *       distribution with mean 0, and std 1.
        ***************************************************************/
        void randn();

        /*Debug Methods*/
        void _printInternalArr() const;



        //Friend class and Functions
        template <typename V>
        friend std::ostream& operator<<(std::ostream& ostr, const Tensor<V> & tensor);

    private:
        T * data;
        int n_els;
        int * mults;
        int * dims;
        int * local_els;
        int n_dims;
        bool contiguous;
        bool track_history = true;
        bool grad_initialized = false;
        Tensor<T> * grad;
        Op<T> * op = NULL;
        std::vector<Tensor*> children;
        std::vector<Tensor*> parents;
};
/*###############################################################################################################*/
/*                                        CONSTRUCTORS/DESTRUCTOR                                                */
/*###############################################################################################################*/
template <typename T>
Tensor<T>::Tensor(int  n_dims, ...) {
    //allocate helpers for indexing
    mults = new int[n_dims];
    dims = new int[n_dims];
    local_els = new int[n_dims];

    this->n_dims = n_dims;

    va_list dimensions;
    va_start(dimensions, n_dims);
    int dim;

    //Copy the dimensions
    for(int i=0; i<n_dims; i++){
        dim = va_arg(dimensions, int);
        this->dims[i] = dim;
    }

    //set the multipliers for each index
    int mult = 1;
    for(int i=n_dims-1; i>=0; i--) {
        mults[i] = mult;
        mult *= dims[i];
        local_els[i] = mult;
    }

    //Clean up variable args
    va_end(dimensions);

    //allocate the full amount of data
    data = new T[mult];
    n_els = mult;

    //Default values
    children = std::vector<Tensor*>();
    parents  =  std::vector<Tensor*>();
    contiguous = true;
}
/*###############################################################################################################*/
template <typename T>
Tensor<T>::Tensor(int n_dims, const int * dims){
    //Allocate the data
    //And copy dims
    int els = 1;
    this->n_dims = n_dims;
    this->dims = new int[n_dims];
    this->mults = new int[n_dims];
    this->local_els = new int[n_dims];
    for(int i=0; i<n_dims; i++) {
        els*=dims[i];
        this->dims[i] = dims[i];
    }
    this->data = new T[els];
    n_els = els;

    //Set the multipliers
    int mult=1;
    for(int i=n_dims-1; i>=0; i--) {
        mults[i] = mult;
        mult *=dims[i];
        local_els[i] = mult;
    }
    n_els = mult;

    //Default values
    children = std::vector<Tensor*>();
    parents  =  std::vector<Tensor*>();
    contiguous = true;
}
/*###############################################################################################################*/
template <typename T>
Tensor<T>::Tensor(const T * data, int n_dims, const int * dims) {
    //Allocate the data
    //And copy dims
    int els = 1;
    this->n_dims = n_dims;
    this->dims = new int[n_dims];
    this->mults = new int[n_dims];
    this->local_els = new int[n_dims];
    for(int i=0; i<n_dims; i++) {
        els*=dims[i];
        this->dims[i] = dims[i];
    }
    this->data = new T[els];
    n_els = els;

    //Copy over the values
    for(int i=0; i<els; i++)
        this->data[i] = data[i];
    
    //Set the multipliers
    int mult=1;
    for(int i=n_dims-1; i>=0; i--) {
        mults[i] = mult;
        mult *=dims[i];
        local_els[i] = mult;
    }
    n_els = mult;

    //Default values
    children = std::vector<Tensor*>();
    parents  =  std::vector<Tensor*>();
    contiguous = true;
}
/*###############################################################################################################*/
template <typename T>
Tensor<T>::Tensor(const Tensor<T>& tensor) {

    //copy values of non pointer values
    this->n_dims = tensor.n_dims;
    this->n_els = tensor.n_els;
    this->contiguous = tensor.contiguous;
    this->children = tensor.children;
    this->parents = tensor.parents;

    this->track_history = tensor.track_history;
    this->grad_initialized = tensor.grad_initialized;
    if(this->grad_initialized){
        //The gradients gradient is not initialized
        //So we wont get stuck in a loop
        this->grad = new Tensor<T>(*tensor.grad);
    }

    //Allocate new memory
    this->data = new T[n_els];
    this->dims = new int[n_dims];
    this->mults = new int[n_dims];
    this->local_els = new int[n_dims];

    //Copy over values
    for(int i=0; i<n_els; i++) 
        this->data[i] = tensor.data[i];
    for(int i=0; i<n_dims; i++) {
        this->dims[i] = tensor.dims[i];
        this->mults[i] = tensor.mults[i];
        this->local_els[i] = tensor.local_els[i];
    }
}
/*###############################################################################################################*/
template <typename T>
Tensor<T> & Tensor<T>::operator=(const Tensor<T>& tensor) {
    if(this != &tensor){
        //copy values of non pointer values
        this->n_dims = tensor.n_dims;
        this->n_els = tensor.n_els;
        this->contiguous = tensor.contiguous;
        this->children = tensor.children;
        this->parents = tensor.parents;

        //Delete and Allocate new memory
        delete [] this->data;
        delete [] this->dims;
        delete [] this->mults;
        delete [] this->local_els;
        
        this->data = new T[n_els];
        this->dims = new int[n_dims];
        this->mults = new int[n_dims];
        this->local_els = new int[n_dims];

        this->track_history = tensor.track_history;
        this->grad_initialized = tensor.grad_initialized;
        if(this->grad_initialized){
            //The gradients gradient is not initialized
            //So we wont get stuck in a loop
            this->grad = new Tensor<T>(*tensor.grad);
        }
        //Copy over values
        for(int i=0; i<n_els; i++) 
            this->data[i] = tensor.data[i];
        for(int i=0; i<n_dims; i++) {
            this->dims[i] = tensor.dims[i];
            this->mults[i] = tensor.mults[i];
            this->local_els[i] = tensor.local_els[i];
        }
    }
    return *this;
}
/*###############################################################################################################*/
template <typename T>
Tensor<T>::~Tensor(){
    /*Have to Deal with Children/Parents when OPS are created*/
    delete [] data;
    delete [] mults;
    delete [] dims;
    delete [] local_els;

    if(op!=NULL) { delete op; }
    if (grad_initialized) delete grad;
}

/*###############################################################################################################*/
/*                                                  Methods                                                      */
/*###############################################################################################################*/

/* The Tensor Methods are moved below the iterator class */
/* So the methods can make use of the iterator class when needed */
template <typename T>
T& Tensor<T>::get(int n_dims , ...) const {
    /*
    Gets the data stored at index dims
    */
    int arr[n_dims];
    va_list indices;
    va_start(indices, n_dims);
    for(int i=0; i<this->n_dims; i++) arr[i] = va_arg(indices, int);
    va_end(indices);

    return get(arr);
}
/*###############################################################################################################*/
template <typename T>
T& Tensor<T>::get(const int * dims) const {
    /*
    Gets the data stored at index dims
    */
    int index = 0;
    for(int i=0; i<n_dims; i++) {
        int sub_index = dims[i];
        index += mults[i] * sub_index;
        assert(dims[i] < this->dims[i] && "OUT OF BOUNDS ERROR");
    }
    assert(index < n_els && "OUT OF BOUNDS ERROR");
    return data[index];
}
/*###############################################################################################################*/
template <typename T>
int Tensor<T>::getIndex(int * dims) const{
    /*
    Returns 1d data array index from the tensor index
    */
    int index = 0;
    for(int i=0; i<n_dims; i++) 
        index += mults[i] * dims[i];
    return index;
}
/*###############################################################################################################*/
template <typename V>    
std::ostream& operator<<(std::ostream& ostr, const Tensor<V> & tensor) {
    /*
    Allows for tensor to be printed to a stream using the 
    << operator
    */
    int curr[tensor.getNDims()];
    for(int i=0; i<tensor.getNDims(); i++) curr[i] = 0;
    iterator<V> it(&tensor, NULL, curr);
    ostr << "DATA: " << std::endl;
    bool opened[tensor.n_dims];
    for(int i=0; i<tensor.n_dims; i++) opened[i] = false;
    for(int i=0; i<tensor.n_els; i++){
        int count = 0;
        for(int j=0; j<tensor.n_dims; j++){
            if ( i%tensor.local_els[j] == 0 && i<tensor.n_els-1 ) {
                ostr << "[";
                count ++;
                opened[j] = true;
            }
        }
        ostr << std::setw(tensor.n_dims+3 - count) << std::setprecision(2) << it.next();

        bool newline = false;
        for(int j=0; j<tensor.n_dims; j++){
            if(opened[j] && (i+1)%tensor.local_els[j] == 0) {
                opened[j] = false;
                newline = true;
                ostr << "]";
            }
        }
        if(newline) ostr << "\n";

    }
    ostr << "METADATA: " << std::endl;
    ostr << "OFFSETS: ";
    for(int i=0; i<tensor.getNDims(); i++) ostr << tensor.getMults()[i] << " ";
    ostr << std::endl;
    ostr << "DIMS: ";
    for(int i=0; i<tensor.getNDims(); i++) ostr << tensor.getDims()[i] << " ";
    ostr << std::endl;
    ostr << "CONTIGUOUS: " << (tensor.is_contiguous() ? "TRUE" : "FALSE") << std::endl;

    ostr << "CHILDREN: " << tensor.children.size() << "\n";
    ostr << "PARENTS: " << tensor.parents.size() << "\n";


    ostr << "GRADIENT: ";
    if(tensor.is_grad_init()){
        ostr << std::endl;
        ostr << "===========================================" << std::endl;
        ostr << "START OF GRADIENT" << std::endl;
        ostr << "===========================================" << std::endl;
        ostr << *tensor.grad << std::endl;
        ostr << "===========================================" << std::endl;
        ostr << "END OF GRADIENT" << std::endl;
        ostr << "===========================================" << std::endl;
    }
    else
        ostr << "Not Initialized" << std::endl;

    return ostr;
}
/*###############################################################################################################*/
template <typename T>
void Tensor<T>::transpose() {
    /*
    Preforms the transpose on the tensor
    if the tensor has n_dims >2
        -it will swap the last 2 dims
    NOTE: THIS WILL THROW OFF THE SHAPE TRACKER, DONT USE UNLESS YOU INTEND TO CORRECT THIS ERRROR
    */
    assert(n_dims >= 2 && "MUST HAVE A SIZE OF AT LEAST 2 TO PREFORM A TRANSPOSE");
    int temp = mults[n_dims-1];
    mults[n_dims-1] = mults[n_dims-2];
    mults[n_dims-2] = temp;
    temp = dims[n_dims-1];
    dims[n_dims-1] = dims[n_dims-2];
    dims[n_dims-2] = temp;
    temp = local_els[n_dims-1];
    local_els[n_dims-1] = local_els[n_dims-2];
    local_els[n_dims-2] = temp;
    contiguous = false;
}
/*###############################################################################################################*/
template <typename T>
void Tensor<T>::as_contiguous() {
    /*
    Rearranges the data array to store
    contiguous values
    */

    //Create new data arr
    T * temp = new T[n_els];

    //set up iterator for tensor
    int curr[n_dims];
    for(int i=0; i<n_dims; i++) curr[i] = 0;
    iterator<T> it(this, NULL,  curr);

    //copy values over
    for(int i=0; i<n_els; i++){
        temp[i] = it.next();
    }

    //recalculate the offsets and local els
    int mult = 1;
    for(int i=n_dims-1; i>=0; i--){
        this->mults[i] = mult;
        mult *= this->dims[i];
        this->local_els[i] = mult;
    }

    //clean up
    contiguous = true;
    delete [] data;
    data = temp;
}
/*###############################################################################################################*/
template <typename T>
void Tensor<T>::_printInternalArr() const {
    for(int i=0; i<n_els; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}
/*###############################################################################################################*/
template <typename T>
void Tensor<T>::reshape(int n_dims, const int * dims) {
    /*
    Reshapes the tensor to have the named dimensions
    */

    int total_els = 1;
    bool done = this->n_dims == n_dims;
    for(int i=0; i<n_dims; i++) {
        if(done){
            done &= (dims[i] == this->dims[i]);
        }
        total_els*=dims[i];
    }
    assert(n_els == total_els && "INVALID SHAPE CONVERSION");

    if(done) return;

    //Make sure the array is contiguous
    if(!contiguous) as_contiguous();

    //Delete previous dimensions
    delete [] this->dims;
    delete [] this->mults;
    delete [] this->local_els;

    //Reallocate metadata arrays
    this->dims = new int[n_dims];
    this->mults = new int[n_dims];
    this->local_els = new int[n_dims];
    this->n_dims = n_dims;
    
    //Copy dims over 
    for(int i=0; i<n_dims; i++) this->dims[i] = dims[i];

    //Calculate new offsets and local_els
    int mult = 1;
    for(int i=n_dims-1; i>=0; i--){
        this->mults[i] = mult;
        mult *= this->dims[i];
        this->local_els[i] = mult;
    }

}
/*###############################################################################################################*/
template <typename T>
void Tensor<T>::reshape(int n_dims, ...) {
    int arr[n_dims];
    va_list dims;
    va_start(dims, n_dims);
    for(int i=0; i<n_dims; i++) arr[i] = va_arg(dims, int);
    va_end(dims);
    reshape(n_dims, arr);
}
/*###############################################################################################################*/
template <typename T>
void Tensor<T>::init_grad() {
    if(grad_initialized) return;
    grad_initialized = true;
    grad = new Tensor<T>(this->n_dims, this->dims);
    grad->setAll(0);
    grad->no_history();
}
/*###############################################################################################################*/
template <typename T>
void Tensor<T>::reshape_grad(int n_dims, const int * dims){
    assert(grad_initialized && "GRAD NOT INITIALIZED");
    int mult = 1;
    for(int i=0; i<n_dims; i++) {
        mult *= dims[i];
    }
    grad->reshape(n_dims, dims);
}
/*###############################################################################################################*/
template <typename T>
void Tensor<T>::randn(){
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0,1);
    for(int i=0; i<n_els; i++){
        data[i] = distribution(generator);
    }
}
#endif
