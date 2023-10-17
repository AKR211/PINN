import numpy as np
from torch import tensor,zeros_like,FloatTensor,linspace,manual_seed,ones_like
from torch.autograd import grad
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt

#Define the architecture of the neural network
class NeuralNet(nn.Module):
	def __init__(self,numinputs=1,numlayers=3,numoutputs=1,numneurons=8):
		super().__init__()
		self.ni = numinputs
		self.no = numoutputs
		self.nl = numlayers
		self.nneu = numneurons

		layers=[]
		layers.append(nn.Linear(self.ni,self.nneu))
		
		for _ in range(self.nl):
			layers.append(nn.Linear(self.nneu,self.nneu))   #create each layer
			layers.append(nn.Tanh())  #tanh activation function is ised
		
		layers.append(nn.Linear(self.nneu,self.no))

		self.Network = nn.Sequential(*layers)    #chain together layers

	def forward(self,input):
		return self.Network(input.view(-1,1))    #forward propogator

#the derivative calculator for nth degree
def dnfdxn(n,model,x_values):
	out = [model(x_values)]
	for i in range(n):
		d = grad(out[-1], x_values, ones_like(out[-1]), create_graph=True)[0]
		out.append(d.view(-1,1))
	return out

#NOW WE SOLVE THE HARMONIC OSCILLATOR DIFFERENTIAL EQUATION

R = 20.0
G=2.0
x_data = tensor([0.,0.1,0.2]).view(-1,1) #sample points
y_data = tensor([0.,0.7478,-0.4984]).view(-1,1)

#creates the loss function using the necessary derivatives
def loss_func_maker(f,dnfdxn):
	
	#The loss function is a sum of MSE loss due to the differetnial equation constraint and the boundary constraints
	def loss_func(x):
		
		f_value, dfdx, d2fdx2 = dnfdxn(2,f,x)
		#f_value = f(x)	
		DEloss = d2fdx2 + 2*G*dfdx + (R**2)*f_value   #loss due to differential equation constraint

		x0 = x_data
		f0 = y_data
		Bdryloss = f(x0) - f0 #loss due to the sample points

		loss = nn.MSELoss()
		loss_val = (1e-4)*loss(DEloss,zeros_like(DEloss)) + loss(Bdryloss,zeros_like(Bdryloss))
		return loss_val

	return loss_func

#for recreating the same output
manual_seed(123)

#define all specifications
inputs=1
outputs=1
layers=2
neurons=32
learning_rate=1e-4
num_epochs=30000
batch_size=30
domain=(0.,1.0)

#we use the adam optimizer

f = NeuralNet(inputs,layers,outputs,neurons)
optimizer = Adam(f.parameters(),lr=learning_rate)
loss_func = loss_func_maker(f,dnfdxn)

x = linspace(domain[0],domain[1],batch_size).view(-1,1).requires_grad_(True)
losses=[]
for i in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_func(x)                         #loss for the batch
    loss.backward()
    optimizer.step()                            #optimization step
    losses.append(float(loss))

X = linspace(domain[0],domain[1],500).reshape(-1,1)   #test data
Y = f(X)
truef = lambda x: np.exp(-2.*x)*	np.sin(19.9*x)         #analytical solution
Y_ = truef(X).view(-1,1)

plt.figure(figsize=(11.8,4.8))
#plot the neural network solution vs the analytical solution
plt.subplot(1,2,1)
plt.plot(X.detach().numpy(),Y.detach().numpy())
plt.plot(X.detach().numpy(),Y_.detach().numpy())
plt.scatter(x_data.detach().numpy(),y_data.detach().numpy())
plt.xlabel('x')
plt.ylabel('f(x)')

#plot the loss with each epoch
plt.subplot(1,2,2)
plt.plot(range(1,num_epochs+1),losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
