import numpy as np
import matplotlib.pyplot as plt

Eb0     =   1.0     # Energy barrier at T0
K       =   0.1     # Thermal conductivity
C       =   10.     # Thermal capacitance
T0      =   10.     # Substrate temperature
Timt    =   30.     # Insulator to metal transition
RI      =   10.     # Insulating resistivity
RM      =   1.      # Metallic resistivity
RL      =   2.      # Load resistivity
N       =   100     # Number of cells

nM      =   np.array(range(0, N))   # Number of metallic cells     
nI      =   N - nM                  # Number of insulating cells
RS      =   nM*RM + nI*RI           # Sample resistance 

def get_T(dt, Vapp):
    VS      =   RS/(RS + RL)*Vapp   # Sample voltage
    Vi      =   VS*RI/RS            # Insulating cell voltage
    Vi      =   VS/nI
    Vi      =   Vapp/nI
    
    return T0 + Vi**2/(RI)*dt
    return T0 + Vi**2/(RI*K)*(1 - np.exp(-K/C*dt))
    
def get_Eb(T):
    Tnorm   =   np.array([t if t < Timt else Timt for t in T])
    return Eb0*(Timt - Tnorm)/(Timt - T0)

def get_P(dt, Vapp, i=N):    
    T       =   get_T(dt, Vapp)
    Eb      =   get_Eb(T)
    
    return np.exp(np.sum(-Eb[:i]/T[:i]))

# Probability distribution

print('Doing probability distribution...')

dtArr       =   [5, 10, 20, 50, 100]        # Observation windows
VappArr     =   np.arange(0., 1.e3, 10.)    # Applied voltages

fig, ax = plt.subplots()

for dt in dtArr:
    PList   =   []
    for Vapp in VappArr:
        PList.append(get_P(dt, Vapp))
    ax.plot(VappArr, PList, label=r'$\Delta t=$'+str(dt))

ax.legend()
plt.savefig('P(Vapp, dt).pdf') 
plt.close()

# Filament length

print('Doing filament length...')

dtArr       =   np.arange(0, 4e3, 1)
VappArr     =   [10., 20., 30., 40., 50.] 

nTrials     =   2

fig, ax = plt.subplots()

for Vapp in VappArr:
    print('Vapp='+str(Vapp))    
    lenFil  =   []      # Filament length     
    for n in range(nTrials):
        print(str(100*n/nTrials)+'% runs', end='\r') 
        r           =   np.random.rand()
        lenFil_n    =   [0]
        for dt in dtArr:
            if lenFil_n[-1] < N:            
                if r < get_P(dt, Vapp, lenFil_n[-1]+1):
                    lenFil_n.append(lenFil_n[-1]+1)                
                else:
                    lenFil_n.append(lenFil_n[-1])
            else:
                lenFil_n = np.concatenate([lenFil_n, lenFil_n[-1]*np.ones(len(dtArr)-len(lenFil_n)+1)])
                break
        lenFil = lenFil_n if len(lenFil) == 0 else np.array(lenFil) + np.array(lenFil_n)  
    lenFil = lenFil / nTrials        
    ax.plot(dtArr, lenFil[1:], label=r'$V_{app}=$'+str(Vapp))
    
ax.legend()
plt.savefig('LenFil(dt, Vapp).pdf') 
plt.close()

# Incubation times

print('Doing incubation times...')

dtArr       =   np.arange(1., 6.e3, 1.)
VappArr     =   np.concatenate([np.arange(10., 100., 10.),
                               np.arange(100., 300., 100.)])
nTrials     =   200   

tIncMeanArr =   []      # Mean incubation times
tIncVarArr  =   []      # Incubation times variance
tIncStdArr  =   []      # Standard deviation

for Vapp in VappArr:
    print('Vapp='+str(Vapp))
    tIncArr =   []
    PArr    =   [get_P(dt, Vapp) for dt in dtArr]
    for i in range(nTrials):
        print(str(100*i/nTrials)+'% runs', end='\r')    
        r = np.random.rand()
        for P, dt in zip(PArr, dtArr):            
            if np.random.rand() < P:
                tIncArr.append(dt)
                break
    tIncArr =   np.array(tIncArr)
    
    # If we observed an event...
    if len(tIncArr) != 0:   
        print('n events='+str(len(tIncArr)))
        
        # Padding
        #tIncArr = np.concatenate([tIncArr, dtArr[-1]*np.ones(nTrials-len(tIncArr))])
        
        tIncMeanArr.append(np.mean(tIncArr))
        tIncVarArr.append(np.var(tIncArr)/len(tIncArr))
        tIncStdArr.append(3*np.sqrt(tIncVarArr[-1]))
        
        #print(tIncArr)        
        print('tInc mean='+str(np.round(tIncMeanArr[-1],2)))
        print('tInc var='+str(np.round(tIncVarArr[-1],2)))
        print('tInc 3*std='+str(np.round(tIncStdArr[-1],2)))
    else:
        VappArr = np.delete(VappArr, 0)

fig, ax = plt.subplots()
ax.plot(VappArr, tIncMeanArr)
ax.errorbar(VappArr, tIncMeanArr, yerr=tIncVarArr)
ax.set_yscale('log')
plt.savefig('tInc(Vapp).pdf') 
plt.close()