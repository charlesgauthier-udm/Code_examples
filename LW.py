###############################################################################
#   LAX-WANDERIEUSLTH
###############################################################################
#==============================================================================
#   IMPORTATION DES MODULES NÉCESSAIRES
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import time
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

def u_x(x,y,t):
# fonction qui calcule la composante x de la vitesse à chq point de la maille
    A = param_vitesse[0]        # amplitude
    ep = param_vitesse[1]       # epsilon param
    phi = param_vitesse[2]      # phase
    w = param_vitesse[3]        # omega
    
    xv,yv = np.meshgrid(x,y)    #Grille de points
    
    ux = A*np.cos(yv+ep*np.sin(w*t+phi))    # tableau composante x vitesse
    return ux

def u_y(x,y,t):
# fonction composante y de la vitesse à chq point de la maille
    B = param_vitesse[0]        # amplitude
    ep = param_vitesse[1]       # epsilon
    phi = param_vitesse[2]      # phase
    w = param_vitesse[3]        # fréquence
    
    xv,yv = np.meshgrid(x,y)    # grille de points
    
    uy = B*np.sin(xv + ep*np.cos(w*t+phi))  # tableau composante y vitesse
    return uy

def source(x,y,t):
# fonction ajout apport source
    x0 = param_source[0]
    y0 = param_source[1]    
    w = param_source[2]
    td = param_source[3]
    tf = param_source[4]
    S0 = param_source[5]
    
    if td <= t <= tf:   # période de contamination
        xv,yv = np.meshgrid(x,y)    # grille de point
        S = S0*np.exp(-((xv-x0)**2 + (yv-y0)**2)/(w**2))    # fct source
    else: S = 0
    
    return S
 
def lw(c,x,y,t):
# fonction qui fait calcule la concentration à un temps donné    
    cnjp1 = np.roll(c,-1,axis=1)            # voisin j+1,k
    cnjm1 = np.roll(c,1,axis=1)             # voisin j-1,k
    cnkp1 = np.roll(c,-1,axis=0)            # voisin j,k+1
    cnkm1 = np.roll(c,1,axis=0)             # voisin j,k-1
    
    cjpkp = np.roll(cnjp1,-1,axis=0)        # voisin j+1,k+1
    cjmkp = np.roll(cnjm1,-1,axis=0)        # voisin j-1,k+1
    cjmkm = np.roll(cnjm1,1,axis=0)         # voisin j-1,k-1
    cjpkm = np.roll(cnjp1,1,axis=0)         # voisin j+1,k-1
    
    # termes intermédiraires
    t1cur = cnjp1+c+cjpkp+cnkp1
    t2cur = cnjp1-c
    t3cur = cnkp1-c
    
    t1cul = c+cnjm1+cnkp1+cjmkp
    t2cul = c-cnjm1
    t3cul = cjmkp-cnjm1
    
    t1cdl = cnkm1+cjmkm+c+cnjm1
    t2cdl = cnkm1-cjmkm
    t3cdl = cnjm1-cjmkm
    
    t1cdr = cjpkm+cnkm1+cnjp1+c
    t2cdr = cjpkm-cnkm1
    t3cdr = c-cnkm1
    
    mix = delx/2
    miy = dely/2
    
    # concentration à mi-pas up/down , left/right
    cur = 0.25*(t1cur[1:N+1,1:N+1]) - \
    (delt/2)*(u_x(x,y,t)*(t2cur[1:N+1,1:N+1])/delx\
     + u_y(x,y,t)*(t3cur[1:N+1,1:N+1])/dely)
    
    cul = 0.25*(t1cul[1:N+1,1:N+1]) - \
    (delt/2)*(u_x(x,y,t)*(t2cul[1:N+1,1:N+1])/delx\
     + u_y(x,y,t)*(t3cul[1:N+1,1:N+1])/dely)
    
    cdl = 0.25*(t1cdl[1:N+1,1:N+1]) - \
    (delt/2)*(u_x(x,y,t)*(t2cdl[1:N+1,1:N+1])/delx\
     + u_y(x,y,t)*(t3cdl[1:N+1,1:N+1])/dely)
    
    cdr = 0.25*(t1cdr[1:N+1,1:N+1]) - \
    (delt/2)*(u_x(x,y,t)*(t2cdr[1:N+1,1:N+1])/delx\
     + u_y(x,y,t)*(t3cdr[1:N+1,1:N+1])/dely)
    
    # calcul des flux    
    
    # flux en x
    Fxur = u_x(x+mix,y+miy,t)*cur
    Fxul = u_x(x-mix,y+miy,t)*cul
    Fxdl = u_x(x-mix,y-miy,t)*cdl
    Fxdr = u_x(x+mix,y-miy,t)*cdr
    
    #flux en y
    Fyur = u_y(x+mix,y+miy,t)*cur
    Fyul = u_y(x-mix,y+miy,t)*cul
    Fydl = u_y(x-mix,y-miy,t)*cdl
    Fydr = u_y(x+mix,y-miy,t)*cdr
    
    t3 = (cnjp1 - 2*c + cnjm1)/(delx**2)    # terme diffusion x
    t4 = (cnkp1 - 2*c + cnkm1)/(dely**2)    # terme diffusion y
    
    cnp1 = np.zeros([N+2,N+2])              # initialisation tableau pas+1
# ÉQUATION LAX-WANDREREJDS 
    cnp1[1:N+1,1:N+1] = c[1:N+1,1:N+1] - ((delt/(2*delx))*(Fxdr-Fxdl+Fxur-Fxul))-((delt/(2*dely))*(Fyur-Fydr+Fyul-Fydl))+ (delt/Pe)*(t3[1:N+1,1:N+1] + t4[1:N+1,1:N+1]) + delt*source(x,y,t)
    
    periodique(N,cnp1)                      # condition périodique
    
    return cnp1

#==============================================================================
#   INITIALISATION DES PARAMÈTRES
#==============================================================================
delx = 5e-03*np.pi               # pas de maille x
dely = 5e-03*np.pi               # pas de maille y


X = np.arange(0,2*np.pi+delx,delx)   # maille x
Y = np.arange(0,2*np.pi+dely,dely)   # maille y
N = len(X)                      # taille de la maille

delt = 1e-03                    # pas de temps


Pe = 1e03                       # nombre pecquelet

# PARAMÈTRES DE LA SOURCE
param_source = [0.8*2*np.pi,    # x0 position x source
                0.6*2*np.pi,    # y0 position y source
                0.25,           # param omega
                1e-02,          # ti temps initial
                1.75,           # tf temps final
                2]              # S_0 taux d'émission

# PARAMÈTRES DE L'ÉCOULEMENT
param_vitesse = [np.sqrt(6),    # amplitude
                 1,             # paramètre epsilon
                 0,             # phi
                 5]             # omega

# INITIALISATION DES TABLEAUX
cnm1 = np.zeros([N+2,N+2])      # tableau pas précédent
c = np.zeros([N+2,N+2])         # tableau pas actuel

r = np.arange(0,3.001,delt)     # pas temporels
compte = 1                      # compteur itérations

p1 = int((N-1)/4)               # indice station 1
p2 = int((N-1)/2)               # indice station 2
p3 = int(3*(N-1)/4)             # indice station 3

st1 = np.zeros(len(r))          # tableau station 1
st2 = np.zeros(len(r))          # tableau station 2
st3 = np.zeros(len(r))          # tableau station 3
#==============================================================================
# BOUCLE TEMPORELLE
#==============================================================================

t0 = time.time()                # chrono start
for i in range(0,len(r)):
    t = r[i]                       # temps
    ap1 = lw(c,X,Y,t) 
    st1[i] = c[p1,p3]
    st2[i] = c[p2,p2]
    st3[i] = c[p3,p1]
    c = ap1

t1 = time.time()
T = t1-t0
print('Temps écoulé: ',T)
#==============================================================================
# AFFICHAGE GRAPHIQUE
#==============================================================================
#%%
lst1 = np.load('lst1.npy')
lst2 = np.load('lst2.npy')
lst3 = np.load('lst3.npy')

plt.plot(r,lst1+0.05,'-m')
plt.plot(r,lst2,'-c')
plt.plot(r,lst3+0.1,'-g')
plt.plot(r,st1+0.05,':m')
plt.plot(r,st2,':c')
plt.plot(r,st3+0.1,':g')
plt.xlim(0,3)
plt.xlabel('temps')
plt.ylabel('concentration')

#XX,YY = np.meshgrid(X,Y)
#plt.contourf(X,Y,c[1:N+1,1:N+1])

#plt.imshow(np.flip(c[1:N+1,1:N+1],axis=0))
#plt.colorbar()