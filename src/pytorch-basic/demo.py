import torch
import numpy as np
import cv2

# print(torch.__version__)

class Pytorch_test():
    def __init__(self) -> None:
        super(Pytorch_test, self).__init__()
        self.rand_tensor = torch.randn(2, 2) * 10
        
        
    def _test_rand(self):
        a = self.rand_tensor
        print(a)
        
    def clamp_test(self):
        print(self.rand_tensor)
        '''
        tensor([[  7.2941, -20.5253],
        [ -1.5648,   6.6860]])
        
        tensor([[5., 2.],
        [2., 5.]])
        
        小于2, 用2替.大于5,用5替换,介于之间的不变
        '''
        print(torch.clamp(self.rand_tensor, 2, 5))  # 2< a < 5
        
    def where_func(self):
        a = torch.randn(4, 4)
        b = torch.randn(4, 4)
        out = torch.where(a > 0.5, a, b)
        print(out)
        
    
    def linespace_func(self):
        a = torch.linspace(1, 16, steps=16)
        # tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.,15., 16.])
        print(a)
        '''
        tensor([[ 1.,  2.,  3.,  4.],
        [ 5.,  6.,  7.,  8.],
        [ 9., 10., 11., 12.],
        [13., 14., 15., 16.]])'''
        a = a.view(4, 4)
        print(a)
        
    def masked_select(self):
        '''
        [-21.0869,  10.8668],
        [ -3.6354,  -9.4531]])
        '''
        print(self.rand_tensor)
        mask = torch.ge(self.rand_tensor, 8)
        '''
        tensor([[False,  True],
        [False, False]])'''
        print(mask)
        out = torch.masked_select(self.rand_tensor, mask=mask)
        '''tensor([10.8668])'''
        print(out)
        
    def concatinate_tensor(self):
        a = torch.ones(2, 4)
        b = torch.zeros(2, 4)
        '''
        tensor([[1., 1., 1., 1., 0., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0., 0.]])'''
                # dim=1 -> 横坐标方向
        out = torch.cat([a, b], dim=1)
        '''
        tensor([[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]])
        '''
        # dim=0 -》纵坐标方向
        out1 = torch.cat([a, b], dim=0)
        print(out, out1)
        
        
    def chunk_func(self):
        a = torch.rand(3, 4) * 100
        '''
        tensor([[49.4458, 11.1552, 82.4468, 34.5927],
                [54.6085, 45.8263, 79.9002, 84.8376],
                [21.8069, 45.7133, 63.0930,  8.5530]])'''
        print(a)
        out = torch.chunk(a, 2, dim=1)
        '''
        tensor([[49.4458, 11.1552],
                [54.6085, 45.8263],
                [21.8069, 45.7133]]), 
        tensor([[82.4468, 34.5927],
                [79.9002, 84.8376],
                [63.0930,  8.5530]])
        '''
        print(out)
        '''
        tensor([[49.4458, 11.1552, 82.4468, 34.5927],
                [54.6085, 45.8263, 79.9002, 84.8376]]), 
        tensor([[21.8069, 45.7133, 63.0930,  8.5530]])
        '''
        out1 = torch.chunk(a, 2, dim=0)
        print(out1)
        
    
    def split_func(self):
        a = torch.rand(10, 4)
        '''
        tensor([[0.5320, 0.2480, 0.8963, 0.8442],
            [0.1971, 0.6465, 0.8220, 0.2574],
            [0.4846, 0.8162, 0.9747, 0.0014],
            [0.0745, 0.3606, 0.4237, 0.2155],
            [0.4486, 0.2898, 0.6636, 0.6347],
            [0.5471, 0.3374, 0.5956, 0.3775],
            [0.6533, 0.4783, 0.4698, 0.1496],
            [0.3857, 0.7052, 0.6359, 0.9115],
            [0.1015, 0.8911, 0.6262, 0.0275],
            [0.2675, 0.7585, 0.5763, 0.2464]]) 
        '''
        out = torch.split(a, 3, dim=0) # same as chunk
        '''
        tensor([[0.5320, 0.2480, 0.8963, 0.8442],
                [0.1971, 0.6465, 0.8220, 0.2574],
                [0.4846, 0.8162, 0.9747, 0.0014]]), 
        tensor([[0.0745, 0.3606, 0.4237, 0.2155],
                [0.4486, 0.2898, 0.6636, 0.6347],
                [0.5471, 0.3374, 0.5956, 0.3775]]),
        tensor([[0.6533, 0.4783, 0.4698, 0.1496],
                [0.3857, 0.7052, 0.6359, 0.9115],
                [0.1015, 0.8911, 0.6262, 0.0275]]), 
        tensor([[0.2675, 0.7585, 0.5763, 0.2464]])
        '''
        print(a, out)
        out1 = torch.split(a, [1, 3, 6], dim=0) # differt from chunk
        '''
        (tensor([[0.0936, 0.3535, 0.4340, 0.1386]]), 
        tensor([[0.2055, 0.5720, 0.5312, 0.8922],
                [0.1165, 0.3523, 0.8343, 0.5243],
                [0.7564, 0.9081, 0.7955, 0.6322]]), 
        tensor([[0.1871, 0.3786, 0.3973, 0.7102],
                [0.8283, 0.4318, 0.6368, 0.2683],
                [0.8593, 0.6647, 0.9418, 0.0928],
                [0.2834, 0.8471, 0.2193, 0.7296],
                [0.3046, 0.2510, 0.8732, 0.4466],
                [0.5065, 0.9473, 0.7427, 0.2864]]))
        '''
        print(out1)
        
test = Pytorch_test()
# test._test_rand()
# test.clamp_test()
# test.where_func()
# test.linespace_func()
# test.masked_select()
# test.concatinate_tensor()
# test.chunk_func()
test.split_func()
        