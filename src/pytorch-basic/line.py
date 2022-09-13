import torch

class Line(torch.autograd.Function):
    def __init__(self, *args, **kwargs):
        super(Line, self).__init__(*args, **kwargs)
       
    @staticmethod
    def forward(ctx, w, x, b):
        ctx.save_for_backward(w, x, b)
        return w*x + b
     
    @staticmethod
    def backward(ctx, grad_out):
        w, x, b = ctx.saved_tensors
        grad_x = grad_out * w
        grad_w = grad_out * x
        grad_b = grad_out * 1
        return grad_w, grad_x, grad_b
    
    
x = torch.rand(2, 2, requires_grad=True)
w = torch.rand(2, 2, requires_grad=True)
b = torch.rand(2, 2, requires_grad=True)

out = Line.apply(w, x, b)
out.backward(torch.ones(2, 2))

print(w, x, b)
print(w.grad, x.grad, b.grad)
    

    
    
        