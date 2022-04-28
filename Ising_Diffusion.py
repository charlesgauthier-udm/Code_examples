# -*- coding: utf-8 -*-
###############################################################################
#   CHARLES GAUTHIER
###############################################################################
#==============================================================================
#   IMPORTATION DES MODULES
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

#==============================================================================
#   VARIABLES GLOBALES
#==============================================================================
N = 64              # taille du réseau
nIter = 500        # nombre itérations
k = 1               # constante Boltzmann
#T0 = 0.17            # température
#H = -0.2            # champ magnétique
#param_bit = [32,32,10,2.5,100,10] # x0,y0,R0,delT,t0,sig
D = 0.01
#==============================================================================
#   DÉFINITION DE FONCTIONS
#==============================================================================
def periodique(N,f):
# fonction qui impose la périodicité
    f[1:N+1,0] = f[1:N+1,N-1]   # périodicité horizontale
    f[1:N+1,N+1] = f[1:N+1,2]
    f[0,1:N+1] = f[N-1,1:N+1]   # périodicité verticale
    f[N+1,1:N+1] = f[2,1:N+1]
    f[0,0] = f[N-1,N-1]         # Les 4 coins
    f[N+1,N+1] = f[2,2]
    f[0,N+1] = f[N-1,2]
    f[N+1,0] = f[2,N-1]
def voisin(spin,i):
# fonction calcul énergie
    v1 = (i[0]+1,i[1])          # voisin down
    v2 = (i[0],i[1]+1)          # voisin droite
    v3 = (i[0]-1,i[1])          # voisin up
    v4 = (i[0],i[1]-1)          # voisin gauche
    
    E = np.zeros_like(spin)     # tableau énergie
    Ep= np.zeros_like(spin)     # tableau énergie prime
    
    # Calcul de l'énergie à partir des 4 voisins
    E[i] = -spin[i]*(spin[v1]+spin[v2]+spin[v3]+spin[v4] + H)
    
    # Calcul de l'énergie prime avec 4 voisins
    Ep[i] = -E[i]

    return E,Ep

def ftcs(c,t,param):
# Calcul concentration avec méthode LeapFrog    
    
    cnjp1 = c[2:N+2,1:N+1]  # 4 voisins
    cnjm1 = c[0:N,1:N+1]
    cnkp1 = c[1:N+1,2:N+2]
    cnkm1 = c[1:N+1,0:N]
    
    termed = D                              # terme diffusion
    cnp1 = np.zeros([N+2,N+2])              # initialisation tableau pas+1

    # paramètre bit
    x0 = param_bit[0]
    y0 = param_bit[1]
    R0 = param_bit[2]
    delT = param_bit[3]
    t0 = param_bit[4]
    sig = param_bit[5]
    # calcul zone chauffage
    terme = (xx-x0)**2 + (yy-y0)**2 # zone chauffage
    i = np.where(terme<R0**2)       # hors zone
    f = np.zeros_like(c)
    f[i] = 1
    cnp1[1:N+1,1:N+1] = c[1:N+1,1:N+1] + termed*(cnkp1 + cnjp1 -4*c[1:N+1,1:N+1] + cnkm1 + cnjm1) + f[1:N+1,1:N+1]*delT*np.exp(-((t-t0)/sig)**2)
    periodique(N,cnp1)                      # condition pÃ©riodique
    
    return cnp1

def prob(E,Ep):
    delE = Ep-E
    p = np.exp(-delE/(k*T))  # probabilité de changement
    p[np.where(p>1)] = 1
    return p
#==============================================================================
#   CODE PRINCIPAL
#==============================================================================
x = np.arange(1,N+1,1)          # tableau x
y = np.arange(1,N+1,1)          # tableau yy

xx,yy = np.meshgrid(x,y)        # réseau

t = np.arange(0,nIter)
# IDENTIFICATION NOEUDS BLANC/NOIR
spin = np.zeros([N+2,N+2])
spin[1:N+1:2,1:N+1:2] = -1
spin[2:N+2:2,2:N+2:2] = -1
spin[2:N+2:2,1:N+1:2] = 1
spin[1:N+1:2,2:N+2:2] = 1
iBlanc = np.where(spin==-1)
iNoir = np.where(spin==1)

fig = plt.figure()              # figure animation
camera = Camera(fig)

#spin[1:N+1,1:N+1] = 1#2*np.random.randint(0,2,size=(N,N))-1 # initialisation spin
#periodique(N,spin)              # conditions périodiques
#bol = np.zeros_like(spin)       # tableau bool
Ti = np.zeros([N+2,N+2])


lpara = [0.05,0.1,0.12,0.14,0.16]
# BOUCLE PARAMÈTRE

for i in range(0,len(lpara)):
    T0 = 0.16
    Ti+=T0
    spin[1:N+1,1:N+1] = 1           # initialisation spin
    periodique(N,spin)              # conditions périodiques
    bol = np.zeros_like(spin)       # tableau bool
    param_bit = [32,32,4,2.5,100,15] # x0,y0,R0,delT,t0,sig
    H = -1
    TabM = np.zeros((nIter,1))
    # BOUCLE TEMPORELLE
    for Iter in range (0,nIter):
        TabM[Iter] = np.mean(spin[1:N+1,1:N+1])
        T = ftcs(Ti,Iter,param_bit)
        
        
        # noeuds blancs
        bol[iBlanc] = 1 # seulement noeuds blanc qui change
        bol[iNoir] = 0
        E,Ep = voisin(spin,iBlanc)  # calcul énergies
        p = prob(E,Ep)         # calcul prob
        
        p*=bol         # seulement noeud blanc qui change

        r = np.random.rand(N+2,N+2)     # nbr aléatoire
        ind = np.where(r<p)         # spin qui flip
        spin[ind]*=-1               # on flip les spins
        
        # noeuds noirs
        bol[iBlanc] = 0
        bol[iNoir] = 1
        E,Ep = voisin(spin,iNoir)
        p=prob(E,Ep)
        p*=bol
        ind = np.where(r<p)
        spin[ind]*=-1 
        periodique(N,spin)
        
        Ti = T
       # if Iter%10 == True:
          #  plt.imshow(spin[1:N+1,1:N+1],vmin=-1,vmax=1)   # affichage spin
          #  plt.text(0.1*N,0.1*N,'t = {}'.format(Iter),color='white') # itération
       # plt.text(0.7*N,0.1*N,'D = {}'.format(D),color='white')
            #camera.snap()
    plt.plot(t,TabM,label='$T0$= {}'.format(lpara[i]))
#animation = camera.animate()
#animation.save('ising.gif', writer = 'imagemagick')

plt.legend(loc=7)
plt.ylabel('Magnétisation')
plt.xlabel('Itération')

