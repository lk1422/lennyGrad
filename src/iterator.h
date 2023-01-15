#ifndef ITERATOR_H_
#define ITERATOR_H_

#include "utils.h"

template <typename T> class Tensor;
/*
    The Iterator Class is used to iterate through the 
    ndarray as if it was using n for loops (one for each dimension). 
    The iterator class is undefined after the tensor class is destroyed
    This allows in order iteration of a non contiguous Tensor.
*/
template <typename T>
class iterator {
    public:
        iterator(const Tensor<T> *  tensor, int * order,  ...);
        iterator(const Tensor<T> *  tensor, int * order, int * curr);
        ~iterator(){ 
            delete [] curr; 
            delete [] dims;
            delete [] order;
        }

        /***************************************************************
        * void next();
        *
        *   Description:
        *       This will return the value being pointed 
        *       to by the iterator and the increment by 1
        *       Will return an Error if next() is called on a
        *       index out of bounds index.
        ***************************************************************/
        T& next();

        /***************************************************************
        * void back();
        *
        *   Description:
        *       This will return the value being pointed 
        *       to by the iterator and the decrement by 1
        *       Will return an Error if next() is called on a
        *       index out of bounds index.
        ***************************************************************/
        T& back();

        /***************************************************************
        * void getCurr(int * curr) const;
        *
        *   Description:
        *       sets the elements of curr to the current index
        *       curr must have the same elements as dimensions
        *       of the tensor being iterated. 
        ***************************************************************/
        void getCurr(int * curr) const;

        /***************************************************************
        * void setCurr(int * curr) const;
        *
        *   Description:
        *       sets the current index of the iterator to curr
        ***************************************************************/
        void setCurr(int * dims);

    private:
        const Tensor<T> * tensor;
        int n_dims;
        int * curr;
        int * order;
        int curr_ind;
        int * dims;
        int n_els;
};
/*###############################################################################################################*/

/*###############################################################################################################*/
/*CONSTRUCTORS*/
/*###############################################################################################################*/
template <typename T>
iterator<T>::iterator(const Tensor<T> * const tensor, int * order,  ...) {
    //Set order
    this->order = new int[tensor->getNDims()];
    if(order==NULL){
        for(int i=0; i<tensor->getNDims(); i++) this->order[i] = i;
    }else{
        copyElements(tensor->getNDims(), this->order, order);
    }
    //Set dimensions according to the order
    const int * temp = tensor->getDims();
    this->dims = new int[tensor->getNDims()];

    for(int i=0; i<tensor->getNDims(); i++) {
        this->dims[this->order[i]] = temp[i];
    }

    //Copy the rest of the values over
    this->tensor = tensor;
    n_dims = tensor->getNDims();
    n_els = tensor->getTotalElements();
    curr = new int[n_dims];
    va_list dims;
    va_start(dims, order);
    for(int i=0; i<n_dims; i++) {
        this->curr[this->order[i]] = va_arg(dims, int);
    }
    curr_ind = tensor->getIndex(curr);

    va_end(dims);
}
/*###############################################################################################################*/
template <typename T>
iterator<T>::iterator(const Tensor<T> * tensor, int * order, int * curr) {
    //Set order
    this->order = new int[tensor->getNDims()];
    if(order==NULL){
        for(int i=0; i<tensor->getNDims(); i++) this->order[i] = i;
    }else{
        copyElements(tensor->getNDims(), this->order, order);
    }
    //Set dimensions according to the order
    const int * temp = tensor->getDims();
    this->dims = new int[tensor->getNDims()];

    for(int i=0; i<tensor->getNDims(); i++) {
        //int index = this->order[i];
        this->dims[this->order[i]] = temp[i];
    }

    //Copy the rest of the values over
    this->tensor = tensor;
    n_els = tensor->getTotalElements();
    n_dims = tensor->getNDims();
    this->curr = new int[n_dims];
    curr_ind = tensor->getIndex(curr);
    for(int i=0; i<n_dims; i++) {
        this->curr[this->order[i]] = curr[i];
    }
}
/*###############################################################################################################*/
/*METHODS*/
/*###############################################################################################################*/
template <typename T>
void iterator<T>::setCurr(int * arr) {
    for(int i=0; i<n_dims; i++) curr[order[i]] = arr[i];
}
/*###############################################################################################################*/
template <typename T>
T& iterator<T>::next() {
    /*
    Returns the current location in the iterator
    then increments the index by 1
    */

    int index[n_dims];
    for(int i=0; i<n_dims; i++) {
        index[i] = curr[order[i]];
    }
    curr_ind = tensor->getIndex(index);

    if(curr_ind >= n_els || curr_ind < 0){
        std::cerr << "OUT OF BOUNDS: Index =" << curr_ind << " Num Elements = " << n_els << std::endl;
        assert(false && "OUT OF BOUNDS");
    }

    //Get Value to return
    //To do this we must swap some values around
    T& return_value = tensor->get(index);

    if(curr_ind == n_els-1) {//If last element leave
        curr_ind++;
        return return_value;
    }

    //Increment iterator
    int inc = n_dims - 1;
    while( (curr[inc] + 1) >= dims[inc] ){
        assert(inc >= 0 && "ERROR GETTTING NEXT");
        curr[inc] = 0;
        inc--;
    }
    curr[inc]++;



    return return_value;
}
/*###############################################################################################################*/
template <typename T>
T& iterator<T>::back(){
    /*
    Returns the current location in the iterator
    then decrements the index by 1
    */
    if(curr_ind >= n_els || curr_ind < 0){
        assert(false && "OUT OF BOUNDS");
    }

    //Calculate index
    int index[n_dims];
    for(int i=0; i<n_dims; i++) index[i] = curr[order[i]];
    T& return_value = tensor->get(index);

    if(curr_ind == n_els-1) {//If last element leave
        curr_ind++;
        return return_value;
    }

    //Increment iterator
    int inc = n_dims-1;
    curr_ind--;
    while( (curr[inc]) == 0 ){
        assert(inc < n_dims && "ERROR GETTING PREVIOUS");
        curr[inc] = dims[inc]-1;
        inc--;
    }
    curr[inc]--;
    curr_ind = tensor->getIndex(curr);

    return return_value;
}
/*###############################################################################################################*/
template <typename T>
void iterator<T>::getCurr(int * curr) const {
    for(int i=0; i<n_dims; i++) curr[i] = this->curr[order[i]];
}
/*###############################################################################################################*/
#endif