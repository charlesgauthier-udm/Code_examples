###############################################################################
#
#   CHARLES GAUTHIER - PROJET 1 
#
###############################################################################
#==============================================================================
#   Importation des modules nécessaires
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from numba import njit,jit
from scipy.signal import find_peaks
import time
#==============================================================================
#   Définition de constantes
#==============================================================================
V_K = -12   # mV
V_Na = 115  # mV
V_L = 10.6  # m
g_K = 36    # ms cm-2
g_Na = 120  # ms cm-2


cm = 1 # Capacitance membranaire nanoF

#==============================================================================
#   Définition de fonctions
#==============================================================================

def alpha_n(V):
   a = (0.1 - 0.01*V)/(np.exp(1 - 0.1*V)-1)     # Eq 1.63
   return a

def alpha_m(V):
    a = (2.5 - 0.1*V)/(np.exp(2.5 - 0.1*V)-1)   # Eq 1.64
    return a

def alpha_h(V):
    a = 0.07*np.exp((-1*V)/20)                  # Eq 1.65
    return a

def beta_n(V):
    a = 0.125*np.exp((-1*V)/80)                 # Eq 1.63
    return a

def beta_m(V):
    a = 4*np.exp((-1*V)/18)                     # Eq 1.64
    return a

def beta_h(V):
    a = 1/(np.exp(3 - 0.1*V)+1)                 # Eq 1.65
    return a

def gn(t0,u,I_a,g_L):
# Fonction calculant le côté droit des équations différentiels
    gV = (1/cm)*(I_a - g_K*(u[1]**4)*(u[0] - V_K) - g_Na*(u[2]**3)*u[3]*(u[0] - V_Na) - g_L*(u[0] - V_L))
    gn = alpha_n(u[0])*(1 - u[1]) - beta_n(u[0])*u[1]
    gm = alpha_m(u[0])*(1 - u[2]) - beta_m(u[0])*u[2]
    gh = alpha_h(u[0])*(1 - u[3]) - beta_h(u[0])*u[3]
    
    out = np.array([gV,gn,gm,gh])
    return out

def rk(h,t0,uu,I,g_L):
# Fonction calculant un pas de runge-kutta ordre 4
     g1=gn(t0,uu,I,g_L)                       # Eq (1.15)
     g2=gn(t0+h/2.,uu+h*g1/2.,I,g_L)          # Eq (1.16)
     g3=gn(t0+h/2.,uu+h*g2/2.,I,g_L)          # Eq (1.17)
     g4=gn(t0+h,uu+h*g3,I,g_L)                # Eq (1.18)
     unew=uu+h/6.*(g1+2.*g2+2.*g3+g4)   # Eq (1.19)
     return unew

def I_n(t,I0,Tau1,Tau2):
    f = 2.718*I0*(t/Tau1)*np.exp(-t/Tau2)
    return f

def cond(j,V):
# Fonction calculant la condition d'équilibre
    if j =='n':
        x = alpha_n(V)/(alpha_n(V) + beta_n(V))
        t = 1/(alpha_n(V) + beta_n(V))
    elif j=='m':
        x = alpha_m(V)/(alpha_m(V) + beta_m(V))
        t = 1/(alpha_m(V) + beta_m(V))
    elif j=='h':
        x = alpha_h(V)/(alpha_h(V) + beta_h(V))
        t = 1/(alpha_h(V) + beta_h(V))
    else: print('Entrée invalide')
    
    return x

def run(I_a,Vect):
# fonction qui calcul pour chaque courant initial
# Vect = [nmax,tol,tfin,V0,courant_continu(T/F)]

    
    nMax = Vect[0]          # Nombre maximal d'ittérations
    eps = 1e-5              # Tolérance
    tol = Vect[1]           # limite du courant continu
    tfin = Vect[2]          # Durée d'intégration
    t = np.zeros(nMax)      # Tableau temps
    u = np.zeros((nMax,4))  # Tableau solution
    V0 = Vect[3]            # Voltage initial
    u[0,:] = np.array([V0,cond('n',V0),
           cond('m',V0),cond('h',V0)])          # Conditions initiales
    nn = 0                  #Compteur d'itérations temporelles
    h = 0.1
    while (t[nn] < tfin) and (nn < nMax-1):       # boucle temporelle
        if t[nn]>=tol and Vect[4]==False:
            I_a = 0
        u1 =rk(h, t[nn],u[nn,:],I_a)            # pas pleine longueur
        u2a=rk(h/2.,t[nn],u[nn,:],I_a)          # premier demi-pas
        u2 =rk(h/2.,t[nn],u2a[:],I_a)           # second demi-pas
        delta=max((abs(u2[0]-u1[0]))/max(abs(u2[0]),
                   abs(u1[0])),abs(u2[1]-u1[1]),abs(u2[2]-u1[2]),
                   abs(u2[3]-u1[3]))        # Eq (1.42) 
        if delta > eps:                     # on rejette
            h/=1.5                          # reduction du pas
        else:                               # on accepte le pas
            nn=nn+1                         # compteur des pas de temps
            t[nn]=t[nn-1]+h                 # le nouveau pas de temps
            u[nn,:]=u2[:]                   # la solution a ce pas
            if delta <= eps/2.: h*=1.5      # on augmente le pas
    
    return t[:nn],u[:nn,0],u[:nn,1],u[:nn,2],u[:nn,3]
    
def plot117(I_a,Vect):
# Traçage et formatage de la figure 1.17

    # textes identifiant les courbes
    t1 = '$I_{a}=8.0'r'\mu A/cm^{2}$'
    t2 = '$I_{a}=7.0'r'\mu A/cm^{2}$'           
    t3 = '$I_{a}=6.9'r'\mu A/cm^{2}$' 
    
    # On affiche les textes sur le graphique
    plt.text(1,106,t1)
    plt.text(6,90,t2)
    plt.text(14,3,t3)          
    
    for i in range(0,len(I_a)):                 # boucle sur les courants I_a
        t,V = run(I_a[i],Vect)                  # fonction pour calculer V et t
        plt.plot(t,V,'-')                       # traçage des figures
    plt.xlabel('t [ms]')                        # titre axe x
    plt.ylabel('V [mV]')                        # titre axe y
    plt.axhline(linestyle=':',c='k')            # ligne pointillé à y=0
    plt.axvline(x=0,c='grey',lw=30,alpha=0.25)  # zone de courant continu
    plt.ylim(-20,120)                           # limites axe y
    plt.xlim(0,25)                              # limites axe x
    plt.show()

def plot116(I_a,Vect):
# Traçage et formatage de la figure 1.16
    t,V,n,m,h = run(I_a,Vect)                 # Calcul de t,V,n,m et h
    
    t1 = '0<t<1ms: $I_{a}=7.0 'r'\mu A/cm^{2}$'  # texte à afficher
    plt.text(8,100,t1)                          # affichage du texte
    plt.xlabel('t [ms]')                        # titre axe x
    plt.ylabel('V [mV]')                        # titre axe y
    plt.axhline(linestyle=':',c='k')            # ligne pointillé à y=0  
    plt.axvline(x=0,c='grey',lw=30,alpha=0.25)  # zone de courant continu
    plt.ylim(-20,120)                           # limites axe y
    plt.xlim(0,25)                              # limites axe x
    plt.plot(t,V,'-k',lw=2)                     # traçage de la courbe V(t)
    plt.twinx()                                 # double axe y
    plt.plot(t,n,'-.k',lw=1)                    # traçage de la courbe n(t)
    plt.axhline(y=n[-1],ls=':',c='k',lw=1)      # traçage n à l'équilibre
    plt.plot(t,m,'--k',lw=1)                    # traçage de la courbe m(t)
    plt.axhline(y=m[-1],ls=':',c='k',lw=1)      # traçage m à l'équilibre
    plt.plot(t,h,'-.k',lw=1)                    # traçage de la courbe h(t)
    plt.axhline(y=h[-1],ls=':',c='k',lw=1)      # traçage h à l'équilibre
    plt.show()

def plot118(I_a,Vect):
# traçage et formatage de la figure 1.18        
    t1,V1,n1,m1,h1 = run(I_a[0],Vect)           # calcul V(t) pour I = 6
    t2,V2,n2,m2,h2 = run(I_a[1],Vect)           # calcul V(t) pour I = 10
    
    plt.plot(t1,V1,'--k',label = '$I_{a}=6.0 'r'\mu A/cm^{2}$') #Traçage V(t)
    plt.plot(t2,V2,'-k',label = '$I_{a}=10.0 'r'\mu A/cm^{2}$') #Traçage V(t)
    plt.axhline(ls=':',c='k')                   # ligne horizontale à y =00
    plt.ylim(-50,150)                           # limite axe y
    plt.xlim(0,100)                             # limite axe x
    plt.xlabel('t [ms]')                        # titre axe x
    plt.ylabel('V [mV]')                        # titre axe y
    plt.legend()                                # ajout légende
    plt.show()

def freq(I_a,Vect):
# fonction qui trouve la fréquence et l'amplitude du mouvement périodique
    t,V,n,m,h = run(I_a,Vect)
    
    im = find_peaks(V,height = 0)   #On trouve le sommet des fonctions
    im = im[0]
    tt = t[im]                      # Valeur temporelle du pic
    tt = tt[1:]                     # On omet le premier pic
    VV = V[im]                      # Amplitude
    VV = VV[1:]                     # On omet la première amplitude
    
    freq = np.mean(1/np.diff(tt))     # Différence entre chaque pic + moyenne
    dfreq = np.std(1/np.diff(tt))     # Déviation standart de la moyenne
    
    amp = np.mean(VV)               # Moyenne des amplitudes de chaque pic
    damp = np.std(VV)               # Déviation standart
    
    return freq,dfreq,amp,damp

def explo1(I0,Tau1,Tau2,g_L):
# fonction qui résous les équations différentielles    
    nMax = 2000                 # Nombre maximal d'ittérations
    eps = 1e-5                  # Tolérance
    tfin = 25                   # Durée d'intégration
    t = np.zeros(nMax)          # Tableau temps
    u = np.zeros((nMax,4))      # Tableau solution
    V0 = 0                      # Voltage initial
    u[0,:] = np.array([V0,cond('n',V0),
           cond('m',V0),cond('h',V0)])          # Conditions initiales
    nn = 0                                      #Compteur d'itérations temporelles
    h = 0.1
    while (t[nn] < tfin) and (nn < nMax-1):     # boucle temporelle
        I_a = I_n(t[nn],I0,Tau1,Tau2)
        u1 =rk(h, t[nn],u[nn,:],I_a,g_L)            # pas pleine longueur
        u2a=rk(h/2.,t[nn],u[nn,:],I_a,g_L)          # premier demi-pas
        u2 =rk(h/2.,t[nn],u2a[:],I_a,g_L)           # second demi-pas
        delta=max((abs(u2[0]-u1[0]))/max(abs(u2[0]),
                   abs(u1[0])),abs(u2[1]-u1[1]),abs(u2[2]-u1[2]),
                   abs(u2[3]-u1[3]))        # Eq (1.42) 
        if delta > eps:                     # on rejette
            h/=1.5                          # reduction du pas
        else:                               # on accepte le pas
            nn=nn+1                         # compteur des pas de temps
            t[nn]=t[nn-1]+h                 # le nouveau pas de temps
                                # on reset le courant à 0
            u[nn,:]=u2[:]                   # la solution a ce pas
            if delta <= eps/2.: h*=1.5      # on augmente le pas
    
    return t[:nn],u[:nn,0]

def Var_tau(Tau1,Tau2,I0):
# fonction qui identifie le courant seuil pour différent Tau

    seuil = np.zeros_like(Tau2)             # tableau I_n0
    temps = np.zeros_like(Tau2)             # tableau temps
    Volt = np.zeros_like(Tau2)
    for k in range(0,len(Tau2)):            # boucle sur tau_d
        t0 = np.zeros_like(I0)              # tableau de temps
        V0 = np.zeros_like(I0)              # Tableau de voltage
        for i in range(0,len(I0)):          # Boucle sur les courants
            t,V = explo1(I0[i],Tau1,Tau2[k])   # Calcul de V(t)
            t=t[np.argmax(V)]               # On trouve le point maximum
            V=V[np.argmax(V)]               
            if V<20:            # Si le pic est sous la tolérance, on rejette
                t = 0
                V = 0
            t0[i] = t           # On ajoute au tableau
            V0[i] = V
            print(i+1,'/',len(I0))          # Affichage itération
            
        I = I0[np.nonzero(t0)]              # On garde le courant seuil
        t0 = t0[np.nonzero(t0)]             # On conserve les points seuils
        V0 = V0[np.nonzero(V0)]
        V0 = V0[np.argmin(V0)]              # Point plus proche du courant seuil
        t0 = t0[np.argmin(V0)] 
        I = I[np.argmin(V0)]                # Courant du courant seuil
        seuil[k] = I                        # Ajout tableau courant
        temps[k] = t0                       # Ajout tableau temps
        Volt[k] = V0                        # Ajout tableau volt
        print('\t PASSAGE: ',k+1,'/',len(Tau2)) # affichage itération
    return seuil,temps,Volt
def gl_Iflop(Tau1,Tau2,I0,gl):
    #g_L = gl
    t0 = np.zeros_like(I0)              # tableau de temps
    V0 = np.zeros_like(I0)              # Tableau de voltage
    for i in range(0,len(I0)):          # Boucle sur les courants
        t,V = explo1(I0[i],Tau1,Tau2)   # Calcul de V(t)
        plt.plot(t,V,'-',label = I0[i])
        t=t[np.argmax(V)]               # On trouve le point maximum
        V=V[np.argmax(V)] 
                      
        if V<20:            # Si le pic est sous la tolérance, on rejette
            t = 0
            V = 0
        t0[i] = t           # On ajoute au tableau
        V0[i] = V
        
        plt.plot(t,V,'or')
        plt.legend()
        print(i+1,'/',len(I0))          # Affichage itération
    I = I0[np.nonzero(t0)]              # On garde le courant seuil
    t0 = t0[np.nonzero(t0)]             # On conserve les points seuils
    V0 = V0[np.nonzero(V0)]
    V0 = V0[np.argmin(V0)]              # Point plus proche du courant seuil
    t0 = t0[np.argmin(V0)] 
    I = I[np.argmin(V0)]                # Courant du courant seuil
    return I

def gl_I(Tau1,Tau2,I0,gl):
    t0 = np.zeros_like(gl)              # tableau de temps
    V0 = np.zeros_like(gl)
    for i in range(0,len(gl)):
        
        t,V = explo1(I0,Tau1,Tau2,gl[i])
        #plt.plot(t,V,'-')
        t = t[np.argmax(V)]
        V = V[np.argmax(V)]
        if V<20:
            t=0
            V=0
        #plt.plot(t,V,'or')
        t0[i] = t
        V0[i]=V
        print(i+1,'/',len(gl))
    G = gl[np.nonzero(t0)]
    t0 = t0[np.nonzero(t0)]
    V0 = V0[np.nonzero(V0)]
    #print(V0)
    #print(G)
    G = G[np.argmin(V0)]
    return G
        
#==============================================================================
#   Code principal
#==============================================================================
#%%
I_a = [6,50]
Vect = [2000,1,100,0,True]
#plot117(I_a)
#plot116(7.0)
#plot118(I_a,Vect)

# Calcul de la fréquence et de l'amplitude pour plusieurs courants
I = np.linspace(7,50,60)
Freq = np.zeros(len(I))             # Tableau fréquences
dFreq = np.zeros(len(I))            # Tableau erreur fréquences
Amp = np.zeros(len(I))              # Tableau amplitude
Damp = np.zeros(len(I))             # Tableau erreur amplitude
for i in range(0,len(I)):
    Freq[i],dFreq[i],Amp[i],Damp[i] = freq(I[i],Vect)
    print(i,'/',len(I))
plt.errorbar(I,Freq,yerr = dFreq, fmt = 'ob',mec='k', capsize=1)
plt.ylabel('Fréquence [$ms^{-1}$]',color = 'blue')
plt.twinx()
plt.errorbar(I,Amp,yerr=Damp,fmt='or',mec='k',capsize=1)
plt.ylabel('Amplitude [mV]',color='red')
plt.xlabel('Courant')
#%% Exploration 1 détermination du courant seuil
Tau1 = 0.5      #ms
Tau2 = 0.5      #ms
I0 = np.linspace(5.671,5.673,2)  #Tableau de courants I_0
coul = ['b','r']
#t = np.zeros_like(I0)   # tableau temps
#V = np.zeros_like(I0)   # tableau voltage
for i in range(0,len(I0)):      #Boucle sur les courants
    
    t,V = explo1(I0[i],Tau1,Tau2)
    plt.plot(t,V,'-',label=r'$I_{{{{0}}}} = {0:.4f}$'.format(I0[i]),c = coul[i])
    plt.legend()
plt.xlim(0,25)
plt.ylim(-20,120)
plt.xlabel('t [ms]')
plt.ylabel('V [mV]')
#%% Exploration 1 - Maladie d'Alzheimer
Tau1 = 0.5
Tau2 = np.linspace(0.5,1.5,5)
I0 = np.zeros(1)
I0+=6.0
gl = np.linspace(0.3,10,10)
II = np.zeros_like(gl)
tt = np.zeros_like(gl)
VV = np.zeros_like(gl)
for i in range(0,len(gl)):
    I_s,Tabt,TabV = Var_tau(Tau1,Tau2,I0,gl[i])
    print(Tabt)
    print('g_L: ',i,'/',len(gl))
'''
X = np.zeros_like(I0)
Y = np.zeros_like(I0)
for i in range(0,len(I0)):
    Tab = explo1(I0[i],Tau1,Tau2)
    x = Tab[0]
    y = Tab[1]
    ind = np.argmax(y)
    x=x[ind]
    y=y[ind]
    if y<20:
        y = 0
        x = 0
    X[i] = x
    Y[i] = y
    plt.plot(Tab[0],Tab[1],'-',label = I0[i])
    plt.plot(x,y,'or')
    plt.legend()
    print(i,'/100')
I = I0[np.nonzero(X)]
X = X[np.nonzero(X)]
Y = Y[np.nonzero(Y)]
Y = Y[np.argmin(Y)]
X = X[np.argmin(Y)]
I = I[np.argmin(Y)]
'''
#%%
#plt.plot(Tau2,I_s,'ob',label = 'Courant seuil',mec='k')
#plt.xlabel(r'$\tau_d$ [ms]')
#plt.ylabel(R'$I_{n0}$  $[\mu A$ $cm^{-2}]$')
#plt.figure()
#plt.subplot(1,2,1)
plt.plot(Tau2,TabV,'or',mec='k')
plt.xlabel(r'$\tau_{d}$ [ms]')
plt.ylabel('V [mv]',color = 'red')
#plt.subplot(1,2,2)
plt.twinx()
plt.plot(Tau2,Tabt,'ob',mec='k')
plt.xlabel(r'$\tau_{d}$ [ms]')
plt.ylabel('t [ms]',color='blue')
#%% Sclérose en plaque - gl en fonction de I
Tau1 = 0.5
Tau2 = np.linspace(0.5,1.5,30)
I0 = 10
gl = np.linspace(0.3,20,100)
G = np.zeros_like(Tau2)
t0 = time.time()
for i in range(0,len(Tau2)):
    G[i] = gl_I(Tau1,Tau2[i],I0,gl)
    print('\t PASSAGE: ',i+1,'/',len(Tau2))
t1 = time.time()
Temps = t1-t0
print('\n TEMPS ÉCOULÉ: ',Temps)
#%% Tracage figure
plt.text(0.55,15,r'$I_{n0}$ = 10 $\mu A$ $cm^{-2}$')
plt.plot(Tau2,G,'or',mec='k')
plt.xlabel(r'$I_{n0}$  $[\mu A$ $cm^{-2}]$')
plt.ylabel('$g_L$ [ms $cm^{-2}$]')
plt.semilogy()
