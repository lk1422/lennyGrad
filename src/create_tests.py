import numpy as np
import random

#Create a file with a bunch of random matrixs
#Create a file with the result of adding multiplying and matrix multiplying them

def stringify_tensor(arr):
    write_str = str(len(arr.shape)) + " "
    shape = [str(i) for i in arr.shape]
    write_str += " ".join(shape) + " "
    for element in np.nditer(arr):
        write_str += str(element) + " "
    write_str = write_str[:-1]
    write_str += "\n"
    return write_str

def create_n_rand_pairs(n):
    tensors = []
    for i in range(n):
        #Create Dims
        shape = [random.randint(1, 10) for j in range(random.randint(2, 5))]
        tensors.append(np.random.randn(*shape))
        tensors.append(np.random.randn(*shape))
    return tensors

def create_n_rand_mults(n):
    tensors = []
    for i in range(n):
        #Create Dims
        n_dims = random.randint(2,5)
        shape = [random.randint(1, 10) for j in range(n_dims)]
        shape2 = shape.copy()
        shape2[-2] = shape[-1]
        shape2[-1] = random.randint(1,10)

        tensors.append(np.random.randn(*shape))
        tensors.append(np.random.randn(*shape2))
    return tensors

def create_tensor_file(file_name, tensors):
    f = open(file_name, 'w+')
    f.write(str(len(tensors)) + "\n")
    for i in range(0,len(tensors)):
        arr = tensors[i]
        f.write(stringify_tensor(arr))
    f.close()

def create_add_file(file_name, tensors):
    f = open(file_name, 'w+')
    f.write(str(len(tensors)//2) + "\n")
    for i in range(0,len(tensors), 2):
        t1 = tensors[i]
        t2 = tensors[i+1]
        t3  = t1 + t2
        f.write(stringify_tensor(t3))
    f.close()

def create_sub_file(file_name, tensors):
    f = open(file_name, 'w+')
    f.write(str(len(tensors)//2) + "\n")
    for i in range(0,len(tensors), 2):
        t1 = tensors[i]
        t2 = tensors[i+1]
        t3  = t1 - t2
        f.write(stringify_tensor(t3))
    f.close()

def create_mult_file(file_name, tensors):
    f = open(file_name, 'w+')
    f.write(str(len(tensors)//2) + "\n")
    for i in range(0,len(tensors), 2):
        t1 = tensors[i]
        t2 = tensors[i+1]
        t3  = t1 * t2
        f.write(stringify_tensor(t3))
    f.close()

def create_div_file(file_name, tensors):
    f = open(file_name, 'w+')
    f.write(str(len(tensors)//2) + "\n")
    for i in range(0,len(tensors), 2):
        t1 = tensors[i]
        t2 = tensors[i+1]
        t3  = t1 / t2
        f.write(stringify_tensor(t3))
    f.close()

def create_matmul_file(file_name, tensors):
    f = open(file_name, 'w+')
    f.write(str(len(tensors)//2) + "\n")
    for i in range(0,len(tensors), 2):
        t1 = tensors[i]
        t2 = tensors[i+1]
        t3  = np.matmul(t1,t2)
        f.write(stringify_tensor(t3))
    f.close()

def create_neg_file(file_name, tensors):
    f = open(file_name, 'w+')
    f.write(str(len(tensors)) + "\n")
    for i in range(0,len(tensors)):
        t1 = tensors[i]
        t3  = -t1
        f.write(stringify_tensor(t3))
    f.close()

def create_exp_file(file_name, tensors):
    f = open(file_name, 'w+')
    f.write(str(len(tensors)) + "\n")
    for i in range(0,len(tensors)):
        t1 = tensors[i]
        t3  = np.exp(t1)
        f.write(stringify_tensor(t3))
    f.close()

def create_relu_file(file_name, tensors):
    f = open(file_name, 'w+')
    f.write(str(len(tensors)) + "\n")
    for i in range(0,len(tensors)):
        t1 = tensors[i]
        t3  = np.maximum(0, t1)
        f.write(stringify_tensor(t3))
    f.close()


if __name__ == "__main__":
    tensors = create_n_rand_pairs(1000)
    create_tensor_file("../testfiles/tensors.txt",tensors)
    create_add_file("../testfiles/add.txt",tensors)
    create_sub_file("../testfiles/sub.txt",tensors)
    create_mult_file("../testfiles/mult.txt",tensors)
    create_div_file("../testfiles/div.txt",tensors)
    create_neg_file("../testfiles/neg.txt",tensors)
    create_exp_file("../testfiles/exp.txt",tensors)
    create_relu_file("../testfiles/relu.txt",tensors)
    m_tensors = create_n_rand_mults(1000)
    create_tensor_file("../testfiles/m_tensors.txt",m_tensors)
    create_matmul_file("../testfiles/matmul.txt", m_tensors)







