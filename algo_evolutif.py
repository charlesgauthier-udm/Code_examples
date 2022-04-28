# -*- coding: utf-8 -*-
###############################################################################
#   CHARLES GAUTHIER PROJET 5
###############################################################################
#==============================================================================
#   IMPORTATION DES MODULES NÉCESSAIRES
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import time
from numba import jit
#==============================================================================
#   VARIABLES GLOBALES
#==============================================================================
Np = 6                      # Nombre de paramètre d'un individu
Nind = 10                  # Nombre d'individus

pc = 0.8                    # probabilité de croisement
pm = 0.1                 # probabilité de mutation
pmm = 0.001

nIter =1000               # nombre maximum d'itération
tol = 1/0.42
tol_mut = 0.1               # Tolérance entre 2 fitness pour mutation adaptative
coul8 = ['#870035','#094414','#600d0d','#686d00','#686b00']
#==============================================================================
#   DÉFINITION DE FONCTION
#==============================================================================
def openfile(nom_fichier):
    Tab = np.loadtxt(nom_fichier,unpack=True)
    return Tab
@jit
def func_E(E,e,P,tj,tau):
    return E - e*np.sin(E) - (2*np.pi/P)*(tj-tau)
@jit
def bissec(x1,x2,epsilon,e,P,tau,tj):
    itermax=40 # nombre maximal d’iterations de la bissection
    delta=x2-x1
    k =0  # compteur d’iteration de la bissection
    while (delta > epsilon) and (k <= itermax):
        xm=0.5*(x1+x2) # essai de racine
        fm=func_E(xm,e,P,tj,tau)
        f2=func_E(x2,e,P,tj,tau)
        if fm*f2 > 0.: # racine a droite
            x2=xm
        else: # racine a gauche
            x1=xm
        delta=x2-x1
        k+=1
    return xm # la racine recherchee
@jit
def initialisation(Np,Bp):
# fonction qui initialise les individus
    para = np.zeros([Np,Nind])      # Tableau d'individus
    for i in range(0,Np):           # initialisation aléatoire
        para[i,:] = np.random.uniform(Bp[i][0],Bp[i][1],size=Nind)
    return para

@jit
def evaluation(para):
# fonction qui calcule la fitness
    return ((np.sin(para[0])/para[0])**2)*((np.sin(para[1])/para[1])**2)

@jit
def evaluation2(Nind_para):
# fonction qui calcule la fitness
    chi2 = np.zeros(Nind)
    for i in range(0,Nind):
        para = Nind_para[:,i]
        P = para[0]
        tau = para[1]
        omega = para[2]
        e = para[3]
        K = para[4]
        V0 = para[5]
        
        epsilon = 1e-5      # précision racine
        Tab_V = np.zeros(len(t))
        # Boucle sur les données
        for j in range(0,len(t)):
            tj = t[j]
            x1 = -10*P           # intervale recherche racine
            x2 = 10*P
            E = scipy.optimize.bisect(func_E,x1,x2,args = (e,P,tj,tau),rtol = epsilon,maxiter = int(1e6))# calcule E
            #E = bissec(x1,x2,epsilon,e,P,tau,tj)
            v = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))   # calcul v eq.5.20
            Tab_V[j] = V0 + K*(np.cos(omega+v)+ e*np.cos(omega))# calcul V eq 5.18
        # CALCUL CHI CARRÉ
        chi2[i] = (1/(len(t)-Np))*np.sum(((V - Tab_V)/sig)**2)
    return 1/chi2
@jit
def sel_roulette(rang):
    r=Nind*(Nind+1)/2. *np.random.uniform() # aleatoire uniforme dans [0,S]
    scumul=0.
    for k in range(0,Nind):
        scumul+=(1.+rang[k])        # some cumulative des rangs
        if scumul >= r: return k       # on a trouve
@jit
def main_func(pop,prang):
# fonction d'avancement d'un pas dans l'algorithme évolutif
    # ÉVALUATION
    fitness = evaluation2(pop)   # évaluation de la fitness
    
    # CLASSIFICATION
    rang = np.argsort(fitness)  # classe meilleurs individus
       
    # BOUCLE REMPLACEMENT NOUVELLE GÉNÉRATION
    Nouv = np.zeros([Np,Nind])
    for i in range(0,Nind,2):
        # SELECTION
        i1=sel_roulette(rang) # un premier parent
        i2=i1
        while (i2==i1):
            i2= sel_roulette(rang) # un second, n’importe qui sauf i1   
        indi1 = pop[:,i1]
        indi2 = pop[:,i2]
        
        # REPRODUCTION
        if np.random.uniform() < pc:            # croisement
            r = np.random.uniform(size = Np)
            nindi1 = r*indi1 + (1-r)*indi2
            nindi2 = (1-r)*indi1 + r*indi2
            for h in range(0,Np):               # mutation
                if np.random.uniform() < pm:
                    disp = (Bp[h][1] - Bp[h][0])/10
                    r = np.random.normal(0.,disp)
                    nindi1[h] = nindi1[h] + a*r
                    if nindi1[h] < Bp[h][0]:
                        nindi1[h] = Bp[h][0] + 0.001*Bp[h][0]
                    if nindi1[h] > Bp[h][1]:
                        nindi1[h] = Bp[h][1] - 0.001*Bp[h][1]
            for g in range(0,Np):               # mutation
                if np.random.uniform() < pm:
                    disp = (Bp[g][1] - Bp[g][0])/2
                    r = np.random.normal(0.,disp)
                    nindi2[g] = nindi2[g] + a*r
                    if nindi2[g] < Bp[g][0]:
                        nindi2[g] = Bp[g][0] + 0.001*Bp[g][0]
                    if nindi2[g] > Bp[g][1]:
                        nindi2[g] = Bp[g][1] - 0.001*Bp[g][1]
        else:
            nindi1 = indi1
            nindi2 = indi2
        Nouv[:,i] = nindi1          # Création nouvelle population
        Nouv[:,i+1] = nindi2
    # ÉVALUATION
    Top = pop[:,rang[Nind-1]]       # meilleur ancienne génération
    new_fitness = evaluation2(Nouv) # nouvelle fitness
    rang = np.argsort(new_fitness)  
    prevfit = evaluation(Nouv)      # fitness génération précédente
    # MUTATION ADAPTATIVE
    if np.abs(np.max(prevfit) - np.max(fitness)) < tol_mut:#test amélioration
        out = 1     # si concluant, on ajoute au compteur
    else: out = 0
    
    Nouv[:,rang[0]] = Top           # remplace pire par ancien meilleur
    
    return Nouv,rang,fitness,out
#==============================================================================
#   CODE PRINCIPAL
#==============================================================================
# INITIALISATION TABLEAUX
Tab = openfile('etaBoo.txt')
t = Tab[0]
th = np.linspace(np.min(t),np.max(t),1000)
V = Tab[1]
sig = 1/Tab[2]
Bp =[(200,800),(t[0],t[0]+800),(0,2*np.pi),(0,1),(0,np.max(V)-np.min(V)),(np.min(V),np.max(V))] # bornes de chaque paramètre d'un individu
#para = np.array([494.2,14299.0,5.7397,0.2626,8.3836,1.0026])
#T1 = time.time()
#fitness,Vth = evaluation2(para)
#T2 = time.time()
#T = T2-T1
#print('temps écoulé: ',T)
#print(fitness)
#plt.plot(th,Vth,'-',c='grey')
#plt.errorbar(t,V,yerr = sig,fmt='.k')
#plt.plot(t[90:],V[90:],'.k')

para = initialisation(Np,Bp)    # initialisation population
para[1] = np.random.uniform(t[0],t[0]+para[0]) # ajustement tau
cond_init = para                # condition initiale
fitness = evaluation2(para)      # fitness initiale
rang = np.argsort(fitness)      # rang initial
Iter = 0                        # compteur
a=1
# BOUCLE AVANCEMENT
yfit = np.zeros(nIter)
XX = np.arange(0,nIter)
T1 = time.time()
compteur = 0
while Iter<nIter:
# BOUCLE D'AVANCEMENT
    Nouv,prang,fit,out = main_func(para,rang)   # avancement d'une génération
    para = Nouv                 # remplacement de la population
    rang = prang
    if np.max(fit) > tol:       # vérification
        break
    Iter+=1
    compteur+=out               # compteur pour mutation adaptative
    pm = 0.1                    # reset probabilité mutation
    a = 1                       # reset amplitude
    if compteur >= 50:      # mutation adaptative
        pm +=0.3
        a=3
        compteur = 0        # reset compteur
T2 = time.time()
T = T2-T1
print('Temps écoulé: ',T)

# COMMANDES GRAPHIQUES
plt.plot(XX,1/yfit,label='Mutation adaptative',c = coul8[2])
plt.xlabel('Génération')
plt.ylabel(r'$\chi^2$')
plt.legend()

