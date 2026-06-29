
"""data.py - synthetic 1D signals: class 0 = sinusoid, class 1 = noise/transient."""
import numpy as np
L=256
def make_synthetic(n=1000,seed=42):
    rng=np.random.default_rng(seed); X=np.zeros((n,L)); y=np.zeros(n,int); t=np.linspace(0,1,L)
    for i in range(n):
        if i%2==0:
            f=rng.uniform(5,20); X[i]=np.sin(2*np.pi*f*t)+rng.normal(0,0.3,L); y[i]=0
        else:
            X[i]=rng.normal(0,1,L); y[i]=1
    return {"X":X,"y":y}
