import numpy as np
import matplotlib.pyplot as plt

Eb0     =   1.0     # Energy barrier at T0
K       =   0.1     # Thermal conductivity
C       =   10.     # Thermal capacitance
T0      =   10.     # Substrate temperature
Timt    =   30.     # Insulator to metal transition
RI      =   10.     # Insulating resistivity
RM      =   1.      # Metallic resistivity
RL      =   5.      # Load resistivity
N       =   100     # Number of cells

nM      =   np.array(range(0, N))   # Number of metallic cells     
nI      =   N - nM                  # Number of insulating cells
RS      =   nM*RM + nI*RI           # Sample resistance 

def get_T(dt, Vapp):
    '''
    VS      =   RS/(RS + RL)*Vapp   # Sample voltage
    Vi      =   VS*RI/RS            # Insulating cell voltage
    Vi      =   VS/nI
    '''
    Vi      =   Vapp/nI
    Vi      =   RI/(RI*nI+RM*nM)*Vapp
    
    return T0 + Vi**2/(RI)*dt
    return T0 + Vi**2/(RI*K)*(1 - np.exp(-K/C*dt))
    
def get_Eb(T):
    Tnorm   =   np.array([t if t < Timt else Timt for t in T])
    return Eb0*(Timt - Tnorm)/(Timt - T0)

def get_P(dt, Vapp, i=N):    
    T       =   get_T(dt, Vapp)
    Eb      =   get_Eb(T)
    
    return np.exp(np.sum(-Eb[:i]/T[:i]))
  
# Print energy barrier
fig = plt.figure(figsize=(2, 2))
ax = fig.add_subplot()

Tarr    =   np.arange(0., 60., 1.)
EbArr   =   get_Eb(Tarr)   

ax.plot(Tarr, EbArr, linewidth=2.6, color='black')
ax.set(xlabel='T (arb. units)', ylabel=r'$\Delta$E')
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16) 
ax.xaxis.set_ticks(np.arange(0, 60, 30))
ax.tick_params(axis='both', which='both', labelsize=16, length=5, width=1)
ax.margins(x=0)
fig.tight_layout()
plt.savefig('DeltaE.pdf') 
plt.clf() 

# Probability distribution

print('Doing probability distribution...')

dtArr       =   [100, 50, 20, 10, 4]        # Observation windows
colors      =   ['green',  'grey',          # Curve colors
                'brown', 'navy', 'purple']
VappArr     =   np.arange(0., 810., 10.)    # Applied voltages

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot()

for dt, c in zip(dtArr, colors[:len(dtArr)]):
    PList   =   []
    for Vapp in VappArr:
        PList.append(get_P(dt, Vapp))
    
    # Voltage is normalized for better plotting
    normVappArr = VappArr/10.
    
    ax.plot(normVappArr, PList, label=r'$\Delta$t='+str(dt), linewidth=2.6, color=c)

ax.set(xlabel=r'$10^{-1}$ V (arb. units)', ylabel='P(V)')
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16) 
ax.tick_params(axis='both', which='both', labelsize=16, length=5, width=1)
ax.legend()
ax.margins(x=0)
plt.legend(prop={'size': 16})  
fig.tight_layout()
plt.savefig('P(Vapp, dt).pdf') 
plt.clf()


# Filament length as a function of voltage

print('Doing filament length as a function of V...')

Narr        =   [50, 100, 150, 200]
dtArr       =   np.arange(0, 3001, 1)
VappArr     =   [50., 20., 10., 5.] 
r_dtArr     =   [np.random.rand() for i in dtArr]
colors      =   ['blueviolet', 'slateblue', 'navy', 'royalblue']
labelsize   =   22
legendsize  =   20

for N in Narr:
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()

    nTrials     =   1
        
    print('N='+str(N)) 
    nM      =   np.array(range(0, N))        
    nI      =   N - nM                  
    RS      =   nM*RM + nI*RI 
    
    for Vapp, c in zip(VappArr, colors[:len(VappArr)]):
        print('Vapp='+str(Vapp))    
        lenFil  =   []      # Filament length     
        for n in range(nTrials):
            print(str(100*n/nTrials)+'% runs', end='\r') 
            #r           =   np.random.rand()
            lenFil_n    =   [0]
            for dt, r in zip(dtArr, r_dtArr):
                if lenFil_n[-1] < N:            
                    if r < get_P(dt, Vapp, lenFil_n[-1]+1):
                        lenFil_n.append(lenFil_n[-1]+1)                
                    else:
                        lenFil_n.append(lenFil_n[-1])
                else:
                    break
            lenFil = np.array(lenFil_n) if len(lenFil) == 0 else np.array(lenFil) + np.array(lenFil_n)  
        lenFil = lenFil / nTrials  
        # Time is normalized for better plotting
        ax.plot(dtArr[:len(lenFil)-1]/100., lenFil[1:], label='V='+str(int(Vapp)), linewidth=2.6, color=c)

    ax.set(xlabel=r'$10^{-2}$ Time (arb. units)', ylabel='Filament length (# cells)')
    ax.xaxis.label.set_size(labelsize)
    ax.margins(x=0)
    ax.yaxis.label.set_size(labelsize) 
    ax.set_ylim([0, 200])
    ax.set_xlim([0, 30])
    ax.tick_params(axis='both', which='both', labelsize=labelsize, length=5, width=1) 
    ax.xaxis.set_ticks(np.arange(0, 31, 10))
    plt.legend(prop={'size': legendsize})
    fig.tight_layout()  
    
    # Add space to the top of the title
    plt.subplots_adjust(top=0.90)    
    plt.title('N='+str(int(N))+' (# of cells)', fontsize=labelsize)
    
    plt.savefig('LenFil(dt, Vapp)_N='+str(N)+'.pdf') 
    plt.clf()

# Filament length as a function of gap size

print('Doing filament length as a function of gap size...')

colors = ['brown', 'red', 'darksalmon', 'sienna', 'sandybrown']

for Vapp in VappArr:
    print('Vapp='+str(Vapp)) 
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()

    nTrials     =   1

    for N, c in zip(Narr, colors[:len(Narr)]):
        print('N='+str(N)) 
        nM      =   np.array(range(0, N))        
        nI      =   N - nM                  
        RS      =   nM*RM + nI*RI 
        lenFil  =   []      # Filament length     
        for n in range(nTrials):
            print(str(100*n/nTrials)+'% runs', end='\r') 
            #r           =   np.random.rand()
            lenFil_n    =   [0]
            for dt, r in zip(dtArr, r_dtArr):
                if lenFil_n[-1] < N:            
                    if r < get_P(dt, Vapp, lenFil_n[-1]+1):
                        lenFil_n.append(lenFil_n[-1]+1)                
                    else:
                        lenFil_n.append(lenFil_n[-1])
                else:
                    break
            lenFil = np.array(lenFil_n) if len(lenFil) == 0 else np.array(lenFil) + np.array(lenFil_n)  
        lenFil = lenFil / nTrials     
        # Time is normalized for better plotting    
        ax.plot(dtArr[:len(lenFil)-1]/100., lenFil[1:], label='N='+str(N), linewidth=2.6, color=c)

    ax.set(xlabel=r'$10^{-2}$ Time (arb. units)', ylabel='Filament length (# cells)')
    ax.xaxis.label.set_size(labelsize)
    ax.margins(x=0)
    ax.yaxis.label.set_size(labelsize) 
    ax.set_ylim([0, 200])
    ax.set_xlim([0, 30])
    ax.tick_params(axis='both', which='both', labelsize=labelsize, length=5, width=1) 
    ax.xaxis.set_ticks(np.arange(0, 31, 10))
    plt.legend(prop={'size': legendsize})
    fig.tight_layout()  
    
    # Add space to the top of the title
    plt.subplots_adjust(top=0.90)    
    plt.title('V='+str(int(Vapp))+' (arb. units)', fontsize=labelsize)
    
    plt.savefig('LenFil(dt, N)_Vapp='+str(int(Vapp))+'.pdf') 
    plt.clf()

# Incubation times

print('Doing incubation times...')

<<<<<<< Updated upstream
fig =   plt.figure(figsize=(3, 6))
ax  =   fig.add_subplot()
=======
fig = plt.figure(figsize=(3, 6))
ax = fig.add_subplot()
>>>>>>> Stashed changes

colors      =   ['brown', 'red', 'darksalmon', 'sienna', 'sandybrown']
colors      =   ['black']
Narr        =   [50, 100, 150, 200]
Narr        =   [100]
dtArr       =   np.arange(1., 6.e3, 1.)
<<<<<<< Updated upstream
=======
VappArr     =   [5., 10., 30., 50., 80., 100., 150., 200. ]
nTrials     =   400   
>>>>>>> Stashed changes

nTrials     =   400   

for N, c in zip(Narr, colors):
    print('N='+str(N))
    
    nM      =   np.array(range(0, N))        
    nI      =   N - nM                  
    RS      =   nM*RM + nI*RI 
        
    tIncMeanArr =   []      # Mean incubation times
    tIncVarArr  =   []      # Incubation times variance
    tIncStdArr  =   []      # Standard deviation

    VappArr     =   np.concatenate([[5., 10, 20.], 
                               np.arange(30., 100., 20.),
                               [125., 150., 200.]])  

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
        
<<<<<<< Updated upstream
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
            
    # Voltage is normalized only for the sake of plotting      
    NormVappArr = VappArr/10.
    
    ax.plot(NormVappArr, tIncMeanArr, color=c, linestyle='-', linewidth='0.5', label='N='+str(N))
    eb = ax.errorbar(NormVappArr, tIncMeanArr, yerr=tIncVarArr, color=c, fmt='.', markersize=8, capsize=8)
    eb[-1][0].set_linestyle('dotted')
    
ax.set(xlabel=r'$10^{-1}$ V (arb. units)', ylabel=r'$\tau_{inc}$ (arb. units)')
ax.xaxis.label.set_size(16)
ax.xaxis.set_ticks(np.arange(0, 22, 5))
ax.set_xlim([-2, 22])
=======
        #print(tIncArr)        
        print('tInc mean='+str(np.round(tIncMeanArr[-1],2)))
        print('tInc var='+str(np.round(tIncVarArr[-1],2)))
        print('tInc 3*std='+str(np.round(tIncStdArr[-1],2)))
    else:
        VappArr = np.delete(VappArr, 0)

ax.plot(VappArr, tIncMeanArr, color='black', linestyle='-', linewidth='0.5')
ax.set(xlabel='V (arb. units)', ylabel=r'$\tau_{inc}$ (arb. units)')
eb = ax.errorbar(VappArr, tIncMeanArr, yerr=tIncVarArr, color='black', fmt='.', markersize=8, capsize=8)
eb[-1][0].set_linestyle('dotted')
ax.xaxis.label.set_size(16)
ax.xaxis.set_ticks(np.arange(0, 220, 60))
>>>>>>> Stashed changes
ax.yaxis.label.set_size(16)
#ax.ticklabel_format(axis='x', style='sci', scilimits=(4,4))
ax.tick_params(axis='both', which='both', labelsize=16, length=5, width=1)
ax.set_yscale('log')

fig.tight_layout()
#plt.legend(prop={'size': 16})
plt.savefig('tInc(Vapp).pdf') 
plt.close()
