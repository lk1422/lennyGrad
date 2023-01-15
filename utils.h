#ifndef UTILS_H_
#define UTILS_H_
template <typename T>
void setAllElements(int n, T * arr, T val){
    for(int i=0; i<n; i++) arr[i] = val;
}

template <typename T>
void copyElements(int n, T * dest, const T * src ) {
    for(int i=0; i<n; i++) dest[i] = src[i];
}
#endif