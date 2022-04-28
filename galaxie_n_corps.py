# -*- coding: utf-8 -*-
###############################################################################
#   CHARLES GAUTHIER
###############################################################################
#==============================================================================
#   IMPORTATION DES MODULES NÉCESSAIRES
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from numba import jit
import time
#==============================================================================
#   VARIABLES GLOBALES
#==============================================================================
Ggrav = 2.277e-7 # G en kpc**3/ M_sol / P**2
N = 1000000 # nombre de particules-etoiles
r_D = 6 # largeur du disque gaussien, en kpc
metoile = 1e6/(N/1e5) # masse d’une particule-etoile
L = 30                  # grandeur domaine en kpc
M = 50
Delta = 2*L/M
N_a = 500 # nombre d’anneaux

# VARIABLE DE TEMPS
ti = 0
tf = 1.5
delt = 0.0002   #incrément de temps

# PARAMÈTRE HALO
sig0 = 1e6
rh = 10
alpha = 2
#==============================================================================
#   DÉFINITION DE FONCTIONS
#==============================================================================
def kcarre(M):
# calcul du tableau des k**2 (Eq. 4.45)
    
    ki = np.fft.fftfreq(M)          #frequence associées à la taille (M)
    ki2d = np.repeat(ki, M)         # version 2D de ki
    ki2d.shape = (M,M)              # reforme en 2D
    kx3d = np.tile(ki2d, M) # version 3D de ki2d (=k_x)
    kx3d.shape = (M, M, M) # reforme en 3D
    kz3d = np.tile(np.transpose(ki2d),M) # transposee 3D de ki2d (=k_z)
    kz3d.shape = (M, M, M) # reforme en 3D
    ky3d = np.rot90(kz3d, 1, axes=(2,1)) # rotation pi/2 dans le plan zy (=k_y)
    k2 = (kx3d**2+ky3d**2+kz3d**2) # k_x**2+k_y**2+k_z**2
    
    return k2/(M/(2 * np.pi))**2 # division par (2L/(2pi))^2

def potgravFFT(M,sigma,Ggrav,metoile,Delta,k2):
# Calcule le potentiel gravitationnel dans le plan z=0 par FFT en 3D

    potFFT = np.zeros((M + 1, M + 1)) # potentiel coins du quadrillage
    sigmaFFT = np.zeros((M, M, M)) # densite en 3D
    sigmaFFT[:,:,0] = sigma # densite 2D a z=0 dans tableau 3D
    potk = np.fft.fftn(sigmaFFT, norm='ortho') # FFT 3D de la densite
    oldpotk = potk[0, 0, 0] # pour eviter division par 0 plus bas
    potk /= k2 # division par k^2 (voir Eq. 4.44)
    potk[0, 0, 0] = oldpotk # eviter la division par k=0
    pot_tmp = np.real(np.fft.ifftn(potk))# partie reelle de la FFT inverse
    p0 = pot_tmp[:,:,0] # potentiel dans le plan z=0
    # valeur sur coins du quadrillage = moyenne sur centres des cellules
    potFFT[1:-1,1:-1]=(p0[0:-1,0:-1]+p0[0:-1,1::]+p0[1::,0:-1]+p0[1::,1::])/4

    return potFFT*(-Ggrav*metoile*Delta) # les constantes physiques
def calc_vrot_r(x,y,vx,vy,L,N_a,dr):
# FONCTION CALCULANT LA COURBE DE ROTATION
    # Calcul de la vitesse rotationnelle moyenne de chaque anneau
    r_a = np.linspace(0, L, N_a) # rayon de chaque anneau
    r = np.sqrt(x**2 + y**2) # rayon de chaque etoile
    iann=(r/dr).astype(int) # indice anneau pour chaque etoile
    vrot=-vx*y/r + vy*x/r # vitesse azimutale de chaque etoile
    vrotMoy=np.zeros(N_a) # tableau vitesse azimutale moyenne
    for i in range(N_a): # boucle sur tous les anneaux
        vrotMoy[i]=np.mean(vrot[np.where(iann==i)]) # moyenne sur anneau i
    # conversion d’unite
    kpc = 30856775814913673 * 1000 # de kpc a m
    P_sol = 7.5e15 # de P_sol a s
    ConversionKmSm1 = (kpc/P_sol)/1000 # Conversion de kpc/P_sol a km/s
    return r_a,vrotMoy*ConversionKmSm1
def sigmaH(sig0,rayon,rh,alpha):
    sigH = sig0/(1-((rayon/rh)**alpha))
    #sigH[int(M/2),int(M/2)] = sig0
    return sigH

#==============================================================================
#   CODE PRINCIPAL
#==============================================================================

x,y,rayon = np.zeros(N),np.zeros(N),np.zeros(N) # initialisation tableaux


deltaM=np.zeros(N_a) # masse dans chaque anneau
Mr = np.zeros(N_a) # masse cumulative sous chaque anneau
Omega = np.zeros(N_a) # vitesse angulaire (Keplerienne)
iann = np.zeros(N,dtype='int') # anneau pour chaque particule
dr = 6*r_D/N_a # largeur radiale des anneaux
x = np.random.normal(0.,r_D,N) # initilisation des position
y = np.random.normal(0.,r_D,N)
rayon = np.sqrt(x**2+y**2) # rayon pour chaque particule
for k in range(0,N):
    iann[k] = int(rayon[k]/dr) # anneau ou tombe chaque particule
    deltaM[iann[k]]+=metoile # incremente masse dans cet anneau
    deltaM[iann[k]]+= sigmaH(sig0,rayon[iann[k]],rh,alpha)*4*np.pi*rayon[iann[k]]
Mr[0] = deltaM[0]
for j in range(1,N_a): # masse cumulative sous chaque anneau
    Mr[j] = Mr[j-1] + deltaM[j]
    Omega[j] = np.sqrt( Ggrav*Mr[j-1]/(j*dr)**3 ) # Eq. (4.30)
Omega[0] = Omega[1]

vx,vy = np.zeros(N),np.zeros(N) # initialisation des vitesses

vx=-Omega[iann]*y # composante x de la vitesse
vy= Omega[iann]*x # composante y de la vitesse


vrot = -vx*y/rayon + vy*x/rayon
vx=-Omega[iann]*y+np.random.normal(0.,vrot*0.05,N)
vy= Omega[iann]*x+np.random.normal(0.,vrot*0.05,N)

sigma=np.zeros([M,M]) # densite M X M cellules, =0
k=np.trunc( (x+L)/Delta ).astype(int) # numero de cellule en x
l=np.trunc( (y+L)/Delta ).astype(int) # numero de cellule en y

ind = np.where(k>=M)
k[ind] = M-1
ind = np.where(l>=M)
l[ind] = M-1

xr = np.linspace(0,L,M)
yr = np.linspace(0,L,M)
xx,yy = np.meshgrid(xr,yr)
Rayon = np.sqrt(xx**2 + yy**2)

for n in range(0,N): # boucle sur les N particules
    sigma[k[n],l[n]]+=metoile # cumul masse dans cellule [k,l]
sigH = sigmaH(sig0,Rayon,rh,alpha)
sigma/=Delta**2 # conversion en densite surfacique
#sigma = sigma + sigH
# Initialisation du potentiel
k2 = kcarre(M)
pot = potgravFFT(M,sigma,Ggrav,metoile,Delta,k2)
ix=np.trunc((x+L)/Delta).astype(int) # coin de la cellule correspondante
iy=np.trunc((y+L)/Delta).astype(int)

ind = np.where(ix>=M)
ix[ind] = M-1
ind = np.where(iy>=M)
iy[ind] = M-1

# Calcul des forces
f_x=-( (pot[ix+1,iy]-pot[ix,iy])+(pot[ix+1,iy+1]-pot[ix,iy+1]) )/(2.*Delta)
f_y=-( (pot[ix,iy+1]-pot[ix,iy])+(pot[ix+1,iy+1]-pot[ix+1,iy]) )/(2.*Delta)

t = np.arange(ti,tf,delt)
tabx = np.zeros([N,3])
taby = np.zeros([N,3])
compteur=0
compteur2 = 0

fig = plt.figure()
camera = Camera(fig)

Pc = int(0.03*N)        # portion des étoiles affichées

# BOUCLE TEMPORELLE
T1 = time.time()
for i in range(0,len(t)):
    x = 0.5*f_x/metoile*delt**2 + vx*delt + x   # Calcul positions
    y = 0.5*f_y/metoile*delt**2 + vy*delt + y
    
    vx = f_x/metoile*delt + vx                  # calcul vitesses
    vy = f_y/metoile*delt + vy  
    
    # conditions limites
    vx[np.where(x>L)]*=-1
    vx[np.where(x<-L)]*=-1
    vy[np.where(y>L)]*=-1
    vy[np.where(y<-L)]*=-1
    x[np.where(x> L)]= 2*L-x[np.where(x> L)]
    x[np.where(x<-L)]=-2*L-x[np.where(x<-L)]
    y[np.where(y> L)]= 2*L-y[np.where(y> L)]
    y[np.where(y<-L)]=-2*L-y[np.where(y<-L)]

    # Calcul potentiel
    if compteur%10==0:
        k=np.trunc( (x+L)/Delta ).astype(int) # numero de cellule en x
        l=np.trunc( (y+L)/Delta ).astype(int) # numero de cellule en y

        ind = np.where(k>=M)
        k[ind] = M-1
        ind = np.where(l>=M)
        l[ind] = M-1
        
        for n in range(0,N): # boucle sur les N particules
            sigma[k[n],l[n]]+=metoile # cumul masse dans cellule [k,l]
        sigma/=Delta**2 # conversion en densite surfacique
        sigma = sigma + sigH
        pot = potgravFFT(M,sigma,Ggrav,metoile,Delta,k2)
        ix=np.trunc((x+L)/Delta).astype(int) # coin de la cellule correspondante
        iy=np.trunc((y+L)/Delta).astype(int)
        
        ind = np.where(ix>=M)
        ix[ind] = M-1
        ind = np.where(iy>=M)
        iy[ind] = M-1
        
        f_x=-( (pot[ix+1,iy]-pot[ix,iy])+(pot[ix+1,iy+1]-pot[ix,iy+1]) )/(2.*Delta)
        f_y=-( (pot[ix,iy+1]-pot[ix,iy])+(pot[ix+1,iy+1]-pot[ix+1,iy]) )/(2.*Delta)
        print(compteur,'/',len(t))
    compteur+=1
'''
    # Animation
    if compteur%100==0:    
        plt.plot(x[1:Pc],y[1:Pc],'.k',ms=1)
        plt.axis('equal')
        plt.text(-40,30,'t = {}'.format(round(t[i],2)),color = 'black')
        plt.xlim(-L,L)
        plt.ylim(-L,L)

        camera.snap()
        compteur2+=1

animation = camera.animate()
'''
T2 = time.time()
delT = T2-T1
print('temps écoulé: ',delT)
r_a,vrotmoy = calc_vrot_r(x,y,vx,vy,L,N_a,dr)
plt.plot(r_a,vrotmoy,'.')
#animation.save('gal.gif', writer = 'imagemagick')