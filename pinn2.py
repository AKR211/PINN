import numpy as np
from torch import tensor,zeros_like,FloatTensor,linspace,manual_seed,mean
import torch.nn as nn
from torch.func import functional_call,grad,vmap
import torchopt
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
		return self.Network(input.reshape(-1,1)).squeeze()    #forward propogator

#This function calculated the derivatives of the forward propogator till a given order
def deriv(model,order):

	#Inorder to calculate higher order derivatives non-destructively, we use the functional API (functorch)
	def f(x,params):
		params_dict = {k:v for k,v in zip(list(dict(model.named_parameters()).keys()),list(params))}
		return functional_call(model,params_dict,(x,))

	derivatives=[]
	derivatives.append(f)
	func = f

	for i in range(order):
		func = grad(f)       #calculate the derivative
		func_map = vmap(func,in_dims=(0,None))     #using a vmap to support batching
		derivatives.append(func_map)

	return derivatives

#NOW WE SOLVE THE LOGISTIC DIFFERENTIAL EQUATION


R = 1.0
x_boundary1 = -np.pi/2
f_boundary1 = -1.0
x_boundary2 = np.pi/2
f_boundary2 = 1.0

truef=lambda x: np.sin(x)         #analytical solution

#creates the loss function using the necessary derivatives
def loss_func_maker(f,df2dx2):
	
	#The loss function is a sum of MSE loss due to the differetnial equation constraint and the boundary constraints
	def loss_func(x,params):
	
		f_value = f(x,params)		
		DEloss = df2dx2(x,params) + R*f_value   #loss due to differetnial equation constraint

		x0 = x_boundary1
		f0 = f_boundary1
		x1 = x_boundary2
		f1 = f_boundary2
		Bdryloss1 = f(tensor([x0]),params) - tensor([f0]) #loss due to the boundary constraint
		Bdryloss2 = f(tensor([x1]),params) - tensor([f1])

		loss = nn.MSELoss()
		loss_val = loss(DEloss,zeros_like(DEloss))/x.shape[0] + loss(Bdryloss1,zeros_like(Bdryloss1)) + loss(Bdryloss2,zeros_like(Bdryloss2))
		return loss_val

	return loss_func

#for recreating the same output
manual_seed(420)

#define all specifications
inputs=1
outputs=1
layers=3
neurons=5
learning_rate=1e-4
num_epochs=10000
batch_size=30
domain=(-5.0,5.0)

#we use the adam optimizer of torchopt module for functional API
optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))

model = NeuralNet(inputs,layers,outputs,neurons)
derivatives = deriv(model,2)
f,dfdx,df2dx2 = derivatives
loss_func = loss_func_maker(f,df2dx2)

params = tuple(model.parameters())
x_data = linspace(-5,5,500).view(-1,1)[0:200:20]


losses=[]
x = FloatTensor(batch_size).uniform_(domain[0],domain[1])       #training data for 1 batch
for i in range(num_epochs):
	y_data = f(x_data,params).view(-1,1)[0:200:20]
	y_true = truef(x_data)
	loss = loss_func(x,params) + mean((y_data-y_true)**2) 	    	#loss for the batch
	params = optimizer.step(loss,params)                            #optimization step
	losses.append(float(loss))

X=linspace(domain[0],domain[1],100).reshape(-1,1)                   #test data
Y=f(X,params)
truef=lambda x: np.sin(x)         #analytical solution
Y_=truef(X)

plt.figure(figsize=(11.8,4.8))
#plot the neural network solution vs the analytical solution
plt.subplot(1,2,1)
plt.plot(X.detach().numpy(),Y.detach().numpy())
plt.plot(X.detach().numpy(),Y_.detach().numpy())
plt.plot(x_data.detach().numpy(),y_true,'r.')
plt.xlabel('x')
plt.ylabel('f(x)')

#plot the loss with each epoch
plt.subplot(1,2,2)
plt.plot(range(1,num_epochs+1),losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
