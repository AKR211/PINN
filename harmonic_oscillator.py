import numpy as np
import torch
from torch import tensor,zeros_like,FloatTensor,linspace,manual_seed,ones_like
from torch.autograd import grad
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt

def truef(x):
	cos = torch.sin(20.0*x)
	return cos

class NeuralNet(nn.Module):
	def __init__(self,numinputs=1,numlayers=3,numoutputs=1,numneurons=8):
		super().__init__()
		self.ni = numinputs
		self.no = numoutputs
		self.nl = numlayers
		self.nneu = numneurons

		layers=[]
		layers.append(nn.Linear(self.ni,self.nneu))
		
		for _ in range(self.nl-1):
			layers.append(nn.Linear(self.nneu,self.nneu))   #create each layer
			layers.append(nn.Tanh())  #tanh activation function is ised
		
		layers.append(nn.Linear(self.nneu,self.no))

		self.Network = nn.Sequential(*layers)    #chain together layers

	def forward(self,input):
		return self.Network(input.view(-1,1))    #forward propogator

def dnfdxn(n,model,x_values):
	out = model(x_values)
	for i in range(n):
		out = grad(out, x_values, ones_like(out), create_graph=True)[0]
	return out.view(-1,1)

R = 20.0
x = torch.linspace(0,1,500).view(-1,1)
y = truef(x).view(-1,1)
x_data = tensor([0.,0.1,0.2]).view(-1,1)
y_data = tensor([0.,0.9093,-0.7568]).view(-1,1)

#creates the loss function using the necessary derivatives
def loss_func_maker(f,dnfdxn):
	
	#The loss function is a sum of MSE loss due to the differetnial equation constraint and the boundary constraints
	def loss_func(x):
	
		f_value = f(x)	
		DEloss = dnfdxn(2,f,x) + (R**2)*f_value   #loss due to differetnial equation constraint

		x0 = x_data
		f0 = y_data
		Bdryloss = f(x0) - f0 #loss due to the boundary constraint

		loss = nn.MSELoss()
		loss_val = (1e-4)*loss(DEloss,zeros_like(DEloss)) + loss(Bdryloss,zeros_like(Bdryloss))
		return loss_val

	return loss_func

x_physics = torch.linspace(0,1,30).view(-1,1).requires_grad_(True)

torch.manual_seed(123)
model = NeuralNet(1,3,1,32)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
loss_func = loss_func_maker(model,dnfdxn)
#x_physics = FloatTensor(30).uniform_(0,1).view(-1,1).requires_grad_(True)
x_physics = torch.linspace(0,1,30).view(-1,1).requires_grad_(True)
for i in range(30000):
    optimizer.zero_grad()
    loss = loss_func(x_physics)
    loss.backward()
    optimizer.step()

X = torch.linspace(0,1,500).reshape(-1,1)
Y = model(X)
plt.plot(x,y.detach().numpy())
plt.plot(X.detach().numpy(),Y.detach().numpy())
plt.scatter(x_data,y_data)
print(x_data,y_data)
plt.show()