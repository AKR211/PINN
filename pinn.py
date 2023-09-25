import numpy as np
from torch import tensor,zeros_like,FloatTensor,linspace,manual_seed
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
x_boundary = 0.0
f_boundary = 0.5

#creates the loss function using the necessary derivatives
def loss_func_maker(f,dfdx):
	
	#The loss function is a sum of MSE loss due to the differetnial equation constraint and the boundary constraints
	def loss_func(x,params):
	
		f_value = f(x,params)		
		DEloss = dfdx(x,params) - R*f_value*(1-f_value)   #loss due to differetnial equation constraint

		x0 = x_boundary
		f0 = f_boundary
		Bdryloss = f(tensor([x0]),params) - tensor([f0]) #loss due to the boundary constraint

		loss = nn.MSELoss()
		loss_val = loss(DEloss,zeros_like(DEloss)) + loss(Bdryloss,zeros_like(Bdryloss))
		return loss_val

	return loss_func

#for recreating the same output
manual_seed(42)

#define all specifications
inputs=1
outputs=1
layers=1
neurons=5
learning_rate=1e-1
num_epochs=100
batch_size=30
domain=(-5.0,5.0)

#we use the adam optimizer of torchopt module for functional API
optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))

model = NeuralNet(inputs,layers,outputs,neurons)
derivatives = deriv(model,1)
f,dfdx = derivatives
loss_func = loss_func_maker(f,dfdx)

params = tuple(model.parameters())

losses=[]
for i in range(num_epochs):
	x = FloatTensor(batch_size).uniform_(domain[0],domain[1])       #training data for 1 batch
	loss = loss_func(x,params)                                      #loss for the batch
	params = optimizer.step(loss,params)                            #optimization step
	losses.append(float(loss))

X=linspace(domain[0],domain[1],100).reshape(-1,1)                   #test data
Y=f(X,params)
truef=lambda x: 1.0/(1.0+(1.0/f_boundary-1.0)*np.exp(-R*x))         #analytical solution
Y_=truef(X)

plt.figure(figsize=(11.8,4.8))
#plot the neural network solution vs the analytical solution
plt.subplot(1,2,1)
plt.plot(X.detach().numpy(),Y.detach().numpy())
plt.plot(X.detach().numpy(),Y_.detach().numpy())
plt.xlabel('x')
plt.ylabel('f(x)')

#plot the loss with each epoch
plt.subplot(1,2,2)
plt.plot(range(1,num_epochs+1),losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
