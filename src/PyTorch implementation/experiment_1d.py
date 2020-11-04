import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from rational import *

# True for using Rational activation function,
# False for using ReLU
UseRational = True

class Net(torch.nn.Module):
    def __init__(self, UseRational):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1,50)
        if UseRational:
            self.R1 = Rational()
        else:
            self.R1 = F.relu
        self.fc2 = torch.nn.Linear(50,50)
        if UseRational:
            self.R2 = Rational()
        else:
            self.R2 = F.relu
        self.fc3 = torch.nn.Linear(50,1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.R1(x)
        x = self.fc2(x)
        x = self.R2(x)
        x = self.fc3(x)
        return x

torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)  # x data (tensor), shape=(100, 1)
y = torch.sin(2*x) + 0.1*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# Define the neural net
net = Net(UseRational)
    
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

BATCH_SIZE = 64
EPOCH = 200

torch_dataset = Data.TensorDataset(x, y)

loader = Data.DataLoader(
    dataset=torch_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=2,)

# start training
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader): # for each training step
        
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        prediction = net(b_x)     # input x and predict based on x

        loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    
    # Compute loss function at the epoch
    prediction = net(x)    
    loss = loss_func(prediction, y)
    print("epoch = %d / %d: loss = %f" % (epoch, EPOCH, loss.item()))

fig, ax = plt.subplots(figsize=(16,10))
plt.cla()
ax.set_title('Regression Analysis - model 3, Batches', fontsize=35)
ax.set_xlabel('Independent variable', fontsize=24)
ax.set_ylabel('Dependent variable', fontsize=24)
ax.set_xlim(-11.0, 13.0)
ax.set_ylim(-1.1, 1.2)
ax.scatter(x.data.numpy(), y.data.numpy(), color = "blue", alpha=0.2)
prediction = net(x)     # input x and predict based on x
ax.scatter(x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
plt.savefig('curve_2_model_3_batches.png')
plt.show()

# Print the network parameter
#for parameter in net.parameters():
#    print(parameter)