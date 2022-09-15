import torch
import re
import numpy as np

from net import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data
ff = open("./housing.data").readlines()
data = []
for item in ff:
    out = re.sub(r"\s{2,}"," ",item).strip()
    data.append(out.split(" "))
    
# print(data)
data = np.array(data).astype(float)
# print(data.shape) # (506, 14)
    
X = data[:, 0: -1] # 所有行，第0列到最后一列（不包含）
Y = data[:, -1]
# print(X.shape)   

X_train = X[0:496, ...]
X_test = X[496:, ...]

Y_train = Y[0:496, ...]
Y_test = Y[496:, ...]
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

net = Net(13, 1)
net = net.to(device=device)

# loss
loss_func = torch.nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# traing
for i in range(10000):
    X_data = torch.tensor(X_train, dtype=torch.float32).to(device=device)
    Y_data = torch.tensor(Y_train, dtype=torch.float32).to(device=device)
    
    pred = net.forward(X_data)
    pred = torch.squeeze(pred)
    
    loss = loss_func(pred, Y_data) * 0.001

    optimizer.zero_grad()
    
    loss.backward()
    optimizer.step() 
    
    print("ite: {}, loss: {}".format(i, loss))
    print(pred[0:10])
    print(Y_data[0:10])   
    
    # test
    
    x_test_data = torch.tensor(X_test, dtype=torch.float32).to(device=device)
    y_test_data = torch.tensor(Y_test, dtype=torch.float32).to(device=device)
    
    pred = net.forward(x_test_data)
    pred = torch.squeeze(pred)
    
    loss_test = loss_func(pred, y_test_data) * 0.001
    
    print("ite:{}, loss_test:{}".format(i, loss_test))
    
torch.save(net, "model/model.pkl")