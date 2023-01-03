#include <iostream>
#include <cstdarg>
#include <iomanip>
#include <cassert>
#include <utility>


/*
TODO:
    fully test all written code
    test reshape
    make reshape take in -1 to fill in expected shape
    make flatten 
    look into tensor splicing
    add a copy methods
    overload operator =
    overload copy constructor
    Refactor to accept both ... and * dims
    Make overflow safe with assert and descriptive error messages

*/

template <typename T>
class Tensor {
    public:
        Tensor(int n_dims, ...);
        Tensor(T * data, int n_dims, int * dims);
        ~Tensor();

        //Methods

        //The default arg does nothing, it is used for stdarg to mount off of
        T& get(int dims, ...) const;
        T& get(int * dims) const;
        bool is_contiguous() const {return contiguous;}

        std::pair<int * , int> shape() const {return std::make_pair(dims, n_dims);}
        std::pair<Tensor * , int> getChildren() const {return std::make_pair(children, n_children);}
        std::pair<Tensor *, int> getParents() const {return std::make_pair(parents, n_parents);}
        int * getMults() const { return mults; }
        int * getDims() const { return dims; } 
        int getNDims() const { return n_dims; }
        int getTotalElements() const { return n_els; }
        int getIndex(int * dims) const;

        /*Reshape Methods*/
        void as_contiguous();
        void reshape(int dims=69, ...);
        void reshape(int * dims);
        void transpose();

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

        /*
        Ops * op
        */

        Tensor * children;
        int n_children;

        Tensor * parents;
        int n_parents;
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
        mults[i] = dims[i];
        mult *= dims[i];
        local_els[i] = mult;
    }

    //Clean up variable args
    va_end(dimensions);

    //allocate the full amount of data
    data = new T[mult];
    n_els = mult;

    //Default values
    children = NULL;
    parents  =  NULL;
    n_children = 0;
    n_parents = 0;
    contiguous = true;
}
/*###############################################################################################################*/
template <typename T>
Tensor<T>::Tensor(T * data, int n_dims, int * dims) {
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

    //Default values
    children = NULL;
    parents  =  NULL;
    n_children = 0;
    n_parents = 0;
    contiguous = true;
}
/*###############################################################################################################*/
template <typename T>
Tensor<T>::~Tensor(){
    /*Have to Deal with Children/Parents when OPS are created*/
    delete [] data;
    delete [] mults;
    delete [] dims;
    delete [] local_els;
}
/*###############################################################################################################*/

/* The Tensor Methods are defined below the Iterator class*/

/*
    The Iterator Class is used to iterate through the 
    ndarray as if it was a for loop. The iterator class
    is undefined after the tensor class is destroyed
*/
template <typename T>
class iterator {
    public:
        iterator(const Tensor<T> *  tensor, ...);
        iterator(const Tensor<T> *  tensor, int * curr);
        ~iterator(){ delete [] curr; }

        T next();
        T back();
    private:
        const Tensor<T> * tensor;
        int n_dims;
        int * curr;
        int curr_ind;
        int * dims;
        int n_els;
};
/*###############################################################################################################*/

/*###############################################################################################################*/
/*CONSTRUCTORS*/
/*###############################################################################################################*/
template <typename T>
iterator<T>::iterator(const Tensor<T> * const tensor, ...) {
    this->tensor = tensor;
    this->dims = tensor->getDims();
    n_dims = tensor->getNDims();
    n_els = tensor->getTotalElements();
    curr = new int[n_dims];
    va_list dims;
    va_start(dims, tensor);
    for(int i=0; i<n_dims; i++) {
        curr[i] = va_arg(dims, int);
    }
    curr_ind = tensor->getIndex(curr);

    va_end(dims);
}
/*###############################################################################################################*/
template <typename T>
iterator<T>::iterator(const Tensor<T> * tensor, int * curr) {
    this->tensor = tensor;
    this->dims = tensor->getDims();
    n_els = tensor->getTotalElements();
    n_dims = tensor->getNDims();
    this->curr = new int[n_dims];
    curr_ind = tensor->getIndex(curr);
    for(int i=0; i<n_dims; i++) {
        this->curr[i] = curr[i];
    }
}
/*###############################################################################################################*/
/*METHODS*/
/*###############################################################################################################*/
template <typename T>
T iterator<T>::next() {
    /*
    Returns the current location in the iterator
    then increments the index by 1
    */

    //Get Value to return
    T return_value = tensor->get(curr);
    //Increment iterator

    //If last element leave
    if(curr_ind == n_els-1) return return_value;
    int inc = n_dims - 1;
    curr_ind++;
    while( (curr[inc] + 1) >= dims[inc] ){
        assert(inc >= 0 && "OUT OF BOUNDS ERROR");
        curr[inc] = 0;
        inc--;
    }
    curr[inc]++;

    return return_value;
}
/*###############################################################################################################*/
template <typename T>
T iterator<T>::back(){
    /*
    Returns the current location in the iterator
    then decrements the index by 1
    */
    T return_value = tensor->get(curr);
    //Increment iterator

    //if first element leave
    if(curr_ind == 0) return return_value;
    int inc = n_dims-1;
    curr_ind--;
    while( (curr[inc]) == 0 ){
        assert(inc < n_dims && "OUT OF BOUNDS ERROR");
        curr[inc] = dims[inc]-1;
        inc--;
    }
    curr[inc]--;

    return return_value;
}
/*###############################################################################################################*/

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
    va_list indices;
    va_start(indices, n_dims);
    int index = 0;
    for(int i=0; i<n_dims; i++) {
        int sub_index = va_arg(indices, int);
        index += mults[i] * sub_index;
    }
    assert(index < n_els && "OUT OF BOUNDS ERROR");

    va_end(indices);
    return data[index];
}
/*###############################################################################################################*/
template <typename T>
T& Tensor<T>::get(int * dims) const {
    /*
    Gets the data stored at index dims
    */
    int index = 0;
    for(int i=0; i<n_dims; i++) {
        int sub_index = dims[i];
        index += mults[i] * sub_index;
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
    for(int i=0; i<n_dims; i++) {
        int sub_index = dims[i];
        index += mults[i] * sub_index;
    }
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
    iterator<V> it(&tensor, curr);
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
        ostr << std::setw(tensor.n_dims+3 - count) << std::setprecision(4) << it.next();

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
    std::cout << "METADATA: " << std::endl;
    std::cout << "OFFSETS: ";
    for(int i=0; i<tensor.getNDims(); i++) std::cout << tensor.getMults()[i] << " ";
    std::cout << std::endl;
    std::cout << "DIMS: ";
    for(int i=0; i<tensor.getNDims(); i++) std::cout << tensor.getDims()[i] << " ";
    std::cout << std::endl;
    std::cout << "CONTIGUOUS: " << (tensor.is_contiguous() ? "TRUE" : "FALSE") << std::endl;

    ostr << "CHILDREN: " << tensor.n_children << "\n";
    ostr << "PARENTS: " << tensor.n_parents << "\n";
    return ostr;
}
/*###############################################################################################################*/
template <typename T>
void Tensor<T>::transpose() {
    /*
    Preforms the transpose on the tensor
    if the tensor has n_dims >2
        -it will swap the last 2 dims
    */
    assert(n_dims >= 2 && "MUST HAVE A SIZE OF AT LEAST 2 TO PREFORM A TRANSPOSE");
    int temp = mults[n_dims-1];
    mults[n_dims-1] = mults[n_dims-2];
    mults[n_dims-2] = temp;
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
    iterator<T> it(this, curr);

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


/*###############################################################################################################*/
/*
    AFTER CHECKING ALL PREIVOUS CODE UNCOMMENT
template <typename T>
void Tensor<T>::reshape(int n_dims, int * dims) {
    Reshapes the tensor to have the named dimensions

    //Make sure the array is contiguous
    as_contiguous();

    //Delete previous dimensions
    delete this->dims;
    delete this->mults;
    delete this->local_els;

    //Reallocate metadata arrays
    this->dims = new int[n_dims];
    this->mutls = new int[n_dims];
    this->local_els = new int[n_dims];
    
    //Copy dims over 
    for(int i=0; i<n_dims; i++) this->dims[i] = dims[i];
    this->n_dims = n_dims;

    //Calculate new offsets and local_els
    int mult = 1
    for(int i=n_dim-1; i>=0; i--){
        this->mults[i] = mult;
        mult *= this->dims[i];
        this->local_els[i] = mults;
    }

}
*/
/*###############################################################################################################*/