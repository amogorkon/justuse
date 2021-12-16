import numpy as np
import warnings
warnings.filterwarnings("ignore") #stop from printing RuntimeWarnings
d=np.double
def t(base,p):
    x=d(1)
    for _ in range(p):x=base**x
    return x
def algo(n,p,max_iters=60,threshold=1e-300):
    f = lambda x:t(x,p)-n
    xl=d(min(n,1))
    xu=d(2) if f(2)>0 else d(max(n,1))
    fl,fu = f(xl),f(xu)
    fr=xr=d(0.0)
    for s in range(max_iters):
        xB=(xu+xl)/2
        xRF=xl if fu==np.inf else (xu*fl-xl*fu)/(fl-fu)
        xr=(xB+xRF)/2 if (s%3!=0 or xu-xl>0.5 or abs(fr)>0.5) else xRF
        fr=f(xr)
        if fr<0:xl,fl=xr,fr
        else:xu,fu=xr,fr
        if xu - xl < threshold or abs(fr) < threshold: 
            return (xu+xl)/2
    return (xu+xl)/2
i = d(input("number: "))
p = int(input("tower degree: "))
print(algo(i,p))