import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 

 

def f(x):
    '''
    Must be a valid probability distribution, e.g. integrate to 1, have values between 0,1 etc...
    '''
    x = np.array([x])
    u = np.array([[1,0]])
    s = np.array([[4,0.1],[2,1]])
    num = np.exp(-0.5*(x-u)@np.linalg.inv(s)@(x-u).T)
    den = np.sqrt((2*np.pi)**len(x)*np.linalg.det(s))
    return ((num/den))

def post_plot(f,x_store):
    plt.figure()
    plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
    plt.scatter(x_store[:,0],x_store[:,1],c='k',marker='.',alpha=0.05)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
    return 

def trace_plot(x_store):
    num = len(x_store[0,:])
    it = np.arange(len(x_store[:,0]))
    fig,axs = plt.subplots(num,num,figsize=(10,num*2),gridspec_kw={'width_ratios': [3, 1]})
    for i in range(num):
        axs[i,0].plot(it,x_store[:,i],c='k',linewidth=0.75,alpha=0.8)
        axs[i,0].grid(alpha=0.5)
        axs[i,0].set_ylabel('$x_'+str(i)+'$')
    axs[-1,0].set_xlabel('iterations')
    for i in range(num):
        axs[i,1].hist(x_store[:,i],100,color='k',alpha=0.8)
        axs[i,1].set_ylabel('$f_{x_'+str(i)+'}$')
        axs[i,1].set_xlabel('$x_'+str(i)+'$')
    plt.tight_layout()
    plt.show()
    return 


def q(x,a_trans):
    return np.random.multivariate_normal(x,a_trans*np.eye(len(x)))

def metropolis_hastings(f,x0,a_trans,iterations,plot=False):
    x_store = np.array([x0])
    acceptance = 0 
    for i in tqdm(range(iterations)):
        x_cand = q(x0,a_trans)
        a = min(1,(f(x_cand)/f(x0)))
        u = np.random.uniform()
        if u < a:
            x0 = x_cand
            acceptance += 1
            x_store = np.append(x_store,[x0],axis=0)
        if i > 0 :
            acceptance_prob = acceptance / i 
            if acceptance_prob > 0.5: 
                a_trans += 0.001 
            else:
                a_trans -= 0.001
    acceptance_prob = acceptance / iterations
    print('TERMINATED CHAIN WITH AN ACCPETANCE PROBABILITY OF ', acceptance_prob)
    if plot == True: 
        if len(x_store[0,:]) != 2:
            trace_plot(x_store)
        else:
            trace_plot(x_store)
            post_plot(f,x_store)
        
    return

x0 = [0,0]
a_trans = 0.75
its = 50000
metropolis_hastings(f,x0,a_trans,its,plot=True)


