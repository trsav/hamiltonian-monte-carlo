import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import autograd.numpy as np 
from autograd import grad 

def f(x):
    '''
    Must be a valid probability distribution, e.g. integrate to 1, have values between 0,1 etc...
    '''
    x = np.array([x])
    u = np.array([[0,0]])
    s = np.array([[1,0.3],[0.1,0.1]])
    num = np.exp(-0.5*(x-u)@np.linalg.inv(s)@(x-u).T)
    den = np.sqrt((2*np.pi)**len(x)*np.linalg.det(s))
    return -np.log(((num/den)))

def post_plot(f,x_store,x_substore,x_rejstore,L,title,trace):
    plt.figure(figsize=(9,6))
    plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
    plt.scatter(x_store[:,0],x_store[:,1],c='k',marker='.',alpha=0.9,label='final samples')
    if trace == True:
        plt.plot(x_substore[:,0],x_substore[:,1],c='k',linewidth=0.95,alpha=0.3,label='intermediate samples')
        # for i in range(int(len(x_rejstore[:,0])/L)):
        #     if i == 0:
        #         plt.plot(x_rejstore[L*i:L*(i+1),0],x_rejstore[L*i:L*(i+1),1],linewidth=0.95,c='r',alpha=0.3,label='rejected samples')
        #     else:
        #         plt.plot(x_rejstore[L*i:L*(i+1),0],x_rejstore[L*i:L*(i+1),1],linewidth=0.95,c='r',alpha=0.3)
    plt.grid(alpha=0.5)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return 

def trace_plot(x_store):
    num = len(x_store[0,:])
    it = np.arange(len(x_store[:,0]))
    it = np.arange(len(x_store[:,0]))
    fig,axs = plt.subplots(num,2,figsize=(10,num*2),gridspec_kw={'width_ratios': [3, 1]})
    for i in range(num):
        axs[i,0].plot(it,x_store[:,i],c='k',linewidth=0.75,alpha=0.8)
        axs[i,0].grid(alpha=0.5)
        axs[i,0].set_ylabel('$x_'+str(i)+'$')
    axs[-1,0].set_xlabel('iterations')
    for i in range(num):
        axs[i,1].hist(x_store[:,i],50,color='k',alpha=0.8)
        axs[i,1].set_ylabel('$f_{x_'+str(i)+'}$')
        axs[i,1].set_xlabel('$x_'+str(i)+'$')
    plt.tight_layout()
    plt.show()
    return 


def q(x,a_trans):
    return np.random.multivariate_normal(x,a_trans*np.eye(len(x)))

def hamiltonian_monte_carlo(f,x0,e,L,iterations,burnin,plot=False,trace=True):
    dim = len(x0)
    x0 = np.array(x0)
    M = np.eye(dim)
    x_store = np.array([x0])
    DU = grad(f)
    acceptance = 0 
    if trace == True:
        x_substore = np.array([x0])
        x_rejstore = np.array([x0])
    momentum = np.array(np.random.multivariate_normal([0 for i in range(dim)],M))
    x0_OG = np.copy(x0)
    for i in tqdm(range(iterations)):
        x0 = x_store[-1,:]
        K_start = 0.5 * momentum.T@np.linalg.inv(M)@momentum
        H_start = f(x0) + K_start
        gradient = DU(x0)
        for j in range(L):
            half_momentum = momentum - 0.5 * e * gradient
            x_new = x0 + e * np.linalg.inv(M)@half_momentum
            new_gradient = DU(x_new)
            momentum_new = half_momentum - 0.5 * e * new_gradient
            momentum = np.copy(momentum_new)
            x0 = np.copy(x_new)
            gradient = np.copy(new_gradient)
            if trace == True:
                x_substore = np.append(x_substore,[x0],axis=0)
        K_new = 0.5 * momentum.T@np.linalg.inv(M)@momentum
        H_new = f(x0) + K_new 
        a = min(1,np.exp(H_new-H_start))
        u = np.random.uniform()
        if u < a: 
            x_store = np.append(x_store,np.array([x0]),axis=0)
            momentum = np.array(np.random.multivariate_normal([0 for i in range(dim)],M))
            acceptance += 1
        else:
            momentum = np.array(np.random.multivariate_normal([0 for i in range(dim)],M))
            if trace == True: 
                x_rejstore = np.append(x_rejstore,x_substore[-L:],axis=0)
                x_substore = x_substore[:-L]
    acceptance_prob = acceptance / iterations
    print('TERMINATED CHAIN WITH AN ACCPETANCE PROBABILITY OF ', acceptance_prob)
    if plot == True: 
        if len(x_store[0,:]) != 2:
            trace_plot(x_store[burnin:,:])
        else:
            trace_plot(x_store[burnin:,:])
            if trace == True:
                x_rejstore = x_rejstore[1:] 
                post_plot(f,x_store[burnin:,:],x_substore[L*burnin:,:],x_rejstore[L*burnin:,:],L,'$\epsilon=$'+str(e)+' $L$='+str(L),trace)
            else:
                post_plot(f,x_store[burnin:,:],[],[],L,'$\epsilon=$'+str(e)+' $L$='+str(L),trace)
    return

x0 = [0.2,2]
e = 0.05
L = 5
its = 50
plot = True
trace = True
burnin = 10
hamiltonian_monte_carlo(f,x0,e,L,its,burnin,plot,trace)


