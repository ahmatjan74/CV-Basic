import torch
import numpy as np
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data
ff = open("housing.data").readlines()
data = []
for item in ff:
    out = re.sub(r"\s{2,}", " ", item).strip()
    data.append(out.split(" "))

data = np.array(data).astype(float)
print(data.shape)

X = data[:, 0: -1]
Y = data[:, -1]

Y_train = Y[0:496, ...]
Y_test = Y[496:, ...]
X_train = X[0:496, ...]
X_test = X[496:, ...]

net = torch.load("model/model.pkl")
loss_func = torch.nn.MSELoss()

x_data = torch.tensor(X_test, dtype=torch.float32).to(device)
y_data = torch.tensor(Y_test, dtype=torch.float32).to(device)

pred = net.forward(x_data)
pred = torch.squeeze(pred)
loss_test = loss_func(pred, y_data) * 0.001
print("loss_test:{}".format(loss_test))
