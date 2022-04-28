# -*- coding: utf-8 -*-
#==============================================================================
#   IMPORTATION DES MODULES NÉCESSAIRES
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import jit
#==============================================================================
#   VARIABLES GLOBALES
#==============================================================================
ni,nh1,nh2,no =13,10,6,1        # nombre d’unites d’entree, interne et de sortie
wih1 =np.zeros([ni,nh1])  # poids des connexions entree vers interne 1
wih2 =np.zeros([nh1,nh2])  # poids des connexions interne 1 vers interne 2 
who =np.zeros([nh2,no])  # poids des connexions interne vers sortie
ivec =np.zeros(ni)      # signal en entree
sh1 =np.zeros(nh1)        # signal des neurones interne1
sh2 =np.zeros(nh2)        # signal des neurones interne2
so =np.zeros(no)        # signal des neurones de sortie
err =np.zeros(no)       # signal d’erreur des neurones de sortie
err_class =np.zeros(no)
deltao=np.zeros(no)     # gradient d’erreur des neurones de sortie
deltah1=np.zeros(nh1)     # gradient d’erreur des neurones internes1
deltah2=np.zeros(nh2)
eta =0.1                # parametre d’apprentissage
#==============================================================================
#   DÉFINITION DE FONCTIONS
#==============================================================================
def openfile(nom_fichier):
    Tab = np.loadtxt(nom_fichier,unpack=True,skiprows=1)
    return Tab
def relu(a):
    return np.maximum(0,a)
def drelu(s):
    if s >0:
        b=1
    else:b=0
    return np.heaviside(s,1)
#@jit
def actv(a):
# fonction d'activation sigmoïde
    return 1./(1.+np.exp(-a))       # Eq. (6.5)
#@jit
def dactv(s):
# dérivée analytique de la fonction sigmoïde
    return s*(1.-s)                 # Eq. (6.19)
#@jit
def ffnn(ivec):
# fonction feed foward
    for ih1 in range(0,nh1):          # couche d’entree a couche interne
        sh1[ih1]=actv(np.sum(wih1[:,ih1]*ivec[:]))            # Eq. (6.2)
    for ih2 in range(0,nh2):
        sh2[ih2]=actv(np.sum(wih2[:,ih2]*sh1[:]))
    for io in range(0,no):          # couche interne a couche de sortie
        so[io]=actv(np.sum(who[:,io]*sh2[:]))            # Eq. (6.4)
    return
#@jit
def backprop(err):

# fonction backpropagation
    deltao=err*dactv(so)       # Eq. (6.20)     
    for io in range(0,no):          # couche de sortie a couche interne
        who[:,io]+=eta*deltao[io]*sh2[:]   # Eq. (6.17) pour les wHO 
        
    for ih2 in range(0,nh2):          # couche interne2 a couche interne1
        deltah2[ih2]=dactv(sh2[ih2])*np.sum(deltao*who[ih2,:])            # Eq. (6.21)
        wih2[:,ih2]+=eta*deltah2[ih2]*sh1[:] # Eq. (6.17) pour les wIH
    for ih1 in range(0,nh1):        # couche interne1 à couhe entrée
        deltah1[ih1]=dactv(sh1[ih1])*np.sum(deltah2*wih2[ih1,:])
        wih1[:,ih1]+=eta*deltah1[ih1]*ivec[:]
    return
#@jit
def randomize(n):
    dumvec=np.zeros(n)
    for k in range(0,n):
        dumvec[k]=np.random.uniform()   # tableau de nombre aleatoires
    return np.argsort(dumvec)           # retourne le tableau de rang
#==============================================================================
#   CODE PRINCIPAL
#==============================================================================
# lecture/initialisation de l’ensemble d’entrainement
Tab = openfile('trainingset.txt')   # importation des données
tset = np.transpose(Tab[:13,:])     # input
for j in range(0,ni):
    tset[:,j]/=np.max(np.abs(tset[:,j]))
oset = Tab[13,:]                    # output
nset =1500 # nombre de membres dans ensemble d’entrainement
tnset = 500
niter =1200 # nombre d’iterations d’entrainement
testset = tset[nset:,:]             # données pour test
testoset = oset[nset:]              # labels pour test
a = np.zeros([tnset,no])
a[:,0] = testoset
testoset = a
tset = tset[:nset,:]
oset =oset[:nset]
rmserr=np.zeros(niter)      # erreur rms d’entrainement
rmserr_t=np.zeros(niter)    # erreur rms test
rms_class=np.zeros(niter)   # erreur classification
a = np.zeros([nset,no])
a[:,0] = oset
oset = a

# initialisation aleatoire des poids
wih1=np.random.uniform(-0.5,0.5,size=(ni,nh1))
wih2=np.random.uniform(-0.5,0.5,size=(nh1,nh2))
who=np.random.uniform(-0.5,0.5,size=(nh2,no))
T1 = time.time()
tab_wih1 = np.zeros([niter,ni,nh1])
tab_wih2 = np.zeros([niter,nh1,nh2])
tab_who = np.zeros([niter,nh2,no])

# BOUCLE D'ENTRAIENEMENT
for iter in range(0,niter):     # boucle sur les iteration d’entrainement
    summ=0.
    rvec=randomize(nset)        # melange des membres

    for itrain in range(0,nset):# boucle sur l’ensemble d’entrainement
        itt=rvec[itrain]        # le membre choisi...
        ivec=tset[itt,:]        # ...et son vecteur d’entree
        ffnn(ivec)              # calcule signal de sortie
        for io in range(0,no):  # signaux d’erreur sur neurones de sortie
            err[io]=oset[itt,io]-so[io]
            summ+=err[io]**2     # cumul pour calcul de l’erreur rms
        #print(err)
        backprop(err)           # retropropagation
    
    if iter%100==False:
        print(iter,'/',niter)
    rmserr[iter]=np.sqrt(summ/nset/no) # erreur rms a cette iteration
    tab_wih1[iter,:,:] = wih1
    tab_wih2[iter,:,:] = wih2
    tab_who[iter,:,:] = who

# BOUCLE TEST
for iter in range(0,niter):
    summ=0.
    summ_class=0.
    rvec=randomize(tnset)
    wih1 = tab_wih1[iter,:,:]
    wih2 = tab_wih2[iter,:,:]
    who = tab_who[iter,:,:]
    for itrain in range(0,tnset):
        itt=rvec[itrain]
        ivec=testset[itt,:]
        ffnn(ivec)
        
        for io in range(0,no):
            err[io]=testoset[itt,io]-so[io]
            err_class[io]=testoset[itt,io]-np.round(so[io])
            summ+=err[io]**2
            summ_class+=np.abs(err_class[io])
    if iter%100==False:
        print(iter,'/',niter)
    rmserr_t[iter]=np.sqrt(summ/tnset/no)
    rms_class[iter]=(summ_class/tnset/no)
    
T2 = time.time()
temps = np.abs(T2-T1)
print('Temps écoulé: ',temps)
t = np.arange(0,niter,1)
plt.plot(t,rmserr,'.k')
plt.plot(t,rmserr_t,'.b')
plt.plot(t,rms_class,'.r')
plt.xlabel('itérations d\'entraînement')
plt.semilogx()
plt.ylabel('root mean square')
