import random
from time import sleep
import matplotlib.pyplot as plt
import numpy as np

m = random.random()
# print(m)

def add_func(a, b):
    return a + b

# print(add_func(1,2))

def print_all(x):
    temp = [a * 10 for a in x]
    for i in range(len(temp)):
        print(temp[i])
        
# print_all([1,2,3,4])

# fig1 = plt.figure(figsize=(5,5))
# x = [a * 2 for a in range(10)]
# y = [b * 4 for b in range(10)]

# plt.plot(x, y)
# plt.xlabel('test X')
# plt.ylabel('related Y')
# plt.show()

class NumpyTest():
    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b
        
    def get_ones(self):
        return np.ones([self.a, self.a]) + np.zeros([self.b, self.b])
    
np_test = NumpyTest(a=3, b=3)
print(np_test.get_ones())