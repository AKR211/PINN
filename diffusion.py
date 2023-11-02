import numpy as np
from torch import tensor,zeros_like,FloatTensor,linspace,manual_seed,ones_like,cartesian_prod,hstack,meshgrid,zeros,ones
from torch.autograd import grad
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

def save_gif_PIL(outfile, files, fps=5, loop=0):
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:], save_all=True, duration=int(1000/fps), loop=loop)

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

	def forward(self,*args):
		input = hstack(args)
		return self.Network(input.view((input.shape[0]*input.shape[1])//self.ni,self.ni))    #forward propogator

#the partial derivative calculator for nth degree (both inputs)
def dnfdxn(n,f_value,x_values):
	out = [f_value]
	for i in range(n):
		d = grad(out[-1], x_values, ones_like(out[-1]), create_graph=True)[0]
		out.append(d.view(-1,1))
	return out

#NOW WE SOLVE THE WAVE DIFFERENTIAL EQUATION

c = .1
'''
x_data = tensor([0.0, 0.0345, 0.069, 0.1034, 0.1379, 0.1724, 0.2069, 0.2414, 0.2759, 0.3103, 0.3448, 0.3793, 0.4138, 0.4483, 0.4828, 0.5172, 0.5517, 0.5862, 0.6207, 0.6552, 0.6897, 0.7241, 0.7586, 0.7931, 0.8276, 0.8621, 0.8966, 0.931, 0.9655, 1.0, 0.0, 0.0345, 0.069, 0.1034, 0.1379, 0.1724, 0.2069, 0.2414, 0.2759, 0.3103, 0.3448, 0.3793, 0.4138, 0.4483, 0.4828, 0.5172, 0.5517, 0.5862, 0.6207, 0.6552, 0.6897, 0.7241, 0.7586, 0.7931, 0.8276, 0.8621, 0.8966, 0.931, 0.9655, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).view(-1,1) #sample points
t_data = tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0345, 0.069, 0.1034, 0.1379, 0.1724, 0.2069, 0.2414, 0.2759, 0.3103, 0.3448, 0.3793, 0.4138, 0.4483, 0.4828, 0.5172, 0.5517, 0.5862, 0.6207, 0.6552, 0.6897, 0.7241, 0.7586, 0.7931, 0.8276, 0.8621, 0.8966, 0.931, 0.9655, 1.0, 0.0, 0.0345, 0.069, 0.1034, 0.1379, 0.1724, 0.2069, 0.2414, 0.2759, 0.3103, 0.3448, 0.3793, 0.4138, 0.4483, 0.4828, 0.5172, 0.5517, 0.5862, 0.6207, 0.6552, 0.6897, 0.7241, 0.7586, 0.7931, 0.8276, 0.8621, 0.8966, 0.931, 0.9655, 1.0]).view(-1,1) #sample points

y_data = tensor([0.0, 0.6363, 0.9817, 0.8785, 0.3737, -0.3019, -0.8395, -0.9934, -0.6932, -0.0762, 0.5756, 0.9644, 0.9123, 0.4433, -0.2284, -0.7956, -0.9992, -0.7461, -0.152, 0.5116, 0.9414, 0.9409, 0.5103, -0.1535, -0.7471, -0.9993, -0.7947, -0.2269, 0.4447, 0.9129,
-0.9129, -0.4447, 0.2269, 0.7947, 0.9993, 0.7471, 0.1535, -0.5103, -0.9409, -0.9414, -0.5116, 0.152, 0.7461, 0.9992, 0.7956, 0.2284, -0.4433, -0.9123, -0.9644, -0.5756, 0.0762, 0.6932, 0.9934, 0.8395, 0.3019, -0.3737, -0.8785, -0.9817, -0.6363, 0.0,
-0.0, -0.6363, -0.9817, -0.8785, -0.3737, 0.3019, 0.8395, 0.9934, 0.6932, 0.0762, -0.5756, -0.9644, -0.9123, -0.4433, 0.2284, 0.7956, 0.9992, 0.7461, 0.152, -0.5116, -0.9414, -0.9409, -0.5103, 0.1535, 0.7471, 0.9993, 0.7947, 0.2269, -0.4447, -0.9129,
0.9129, 0.4447, -0.2269, -0.7947, -0.9993, -0.7471, -0.1535, 0.5103, 0.9409, 0.9414, 0.5116, -0.152, -0.7461, -0.9992, -0.7956, -0.2284, 0.4433, 0.9123, 0.9644, 0.5756, -0.0762, -0.6932, -0.9934, -0.8395, -0.3019, 0.3737, 0.8785, 0.9817, 0.6363, 0.0]
).view(-1,1)
'''
truef = lambda x,t: np.exp(-100.0*((x - 0.5)**2))
n=30
x_data = hstack((linspace(0,1,n),zeros(n),ones(n))).view(-1,1)
t_data = hstack((zeros(n),linspace(0,1,n),linspace(0,1,n))).view(-1,1)

y_data = truef(x_data,t_data).view(-1,1)

#creates the loss function using the necessary derivatives
def loss_func_maker(f,dnfdxn):
	
	#The loss function is a sum of MSE loss due to the differetnial equation constraint and the boundary constraints
	def loss_func(x, t):

		f_value = f(x, t)
		f_value, dfdx, d2fdx2 = dnfdxn(2,f_value, x)
		f_value, dfdt, d2fdt2 = dnfdxn(2,f_value, t)
		DEloss = dfdt - (c**2)*d2fdx2   #loss due to differential equation constraint

		x0 = x_data
		t0 = t_data
		f0 = y_data
		Bdryloss = f(x0,t0) - f0 #loss due to the sample points

		loss = nn.MSELoss()
		loss_val = (1e-3)*loss(DEloss,zeros_like(DEloss)) + loss(Bdryloss,zeros_like(Bdryloss))
		return loss_val

	return loss_func

#for recreating the same output
manual_seed(123)

#define all specifications
inputs=2
outputs=1
layers=2
neurons=32
learning_rate=1e-4
num_epochs=10000
batch_size=30
domainx=(0.,1.0)
domainy=(0.,1.0)

#we use the adam optimizer

f = NeuralNet(inputs,layers,outputs,neurons)
optimizer = Adam(f.parameters(),lr=learning_rate)
loss_func = loss_func_maker(f,dnfdxn)

x_ = linspace(domainx[0],domainx[1],batch_size).requires_grad_(True)
t_ = linspace(domainy[0],domainy[1],batch_size).requires_grad_(True)
x,t = meshgrid(x_, t_)
x,t = x.reshape(-1,1) , t.reshape(-1,1)
losses=[]
for i in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_func(x, t)                         #loss for the batch
    loss.backward()
    optimizer.step()                            #optimization step
    losses.append(float(loss))
    if not i%1000:
       print(i, float(loss))

print(f'Final loss is {losses[-1]}')

X_ = linspace(domainx[0],domainx[1],500)   #test data
T_ = linspace(domainy[0],domainy[1],500)   #test data
X,T = meshgrid(X_, T_)
X,T = X.reshape(-1,1) , T.reshape(-1,1)
Y = f(X, T)
truef = lambda x,t: np.sin(20.0*x - 20.0*t)      #analytical solution
Y_ = truef(X, T).view(-1,1)

#plt.figure(figsize=(11.8,4.8))
#plot the neural network solution vs the analytical solution
'''
plt.subplot(1,2,1)

plt.plot(X.detach().numpy(),Y.detach().numpy())
plt.plot(X.detach().numpy(),Y_.detach().numpy())
plt.scatter(x_data.detach().numpy(),y_data.detach().numpy())
plt.xlabel('x')
plt.ylabel('f(x)')

plt.imshow(Y.detach().numpy().reshape(500,500),cmap='Greys', origin='lower')

#plot the loss with each epoch
plt.subplot(1,2,2)
plt.plot(range(1,num_epochs+1),losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
'''
files=[]
i=0
for t in linspace(0.,0.297*np.pi,99):
	x_gif = linspace(0.,1.,500).view(-1,1)
	t_gif = tensor([t]).expand(500,1)
	
	print(i)
	plt.figure(figsize=(8,4))
	plt.plot(x_gif.detach().numpy(),f(x_gif,t_gif).detach().numpy())
	plt.xlim(0.,1.)
	plt.ylim(-0.2,1.2)
	plt.title(f't = {round(t.item(),3)}')
	plt.xlabel('x')
	plt.ylabel('f(x,t)')
	file = "plots/pinn_%.8i.png"%(i+1)
	plt.savefig(file, bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor="white")
	files.append(file)
	plt.close('all')
	i+=1
	
save_gif_PIL("pinn.gif", files, fps=20, loop=0)