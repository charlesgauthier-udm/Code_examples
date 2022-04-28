# rmatctem function script
import numpy as np
from numba import jit

@jit
def unfrozenroot(maxannualactlyr, zbotw, rmatctem):
    iccp2 = 11
    ignd = 20
    ndays = rmatctem.shape[0]
    ncell = rmatctem.shape[3]
    unfrozenroot = np.zeros((ndays, iccp2,ignd,ncell))
    botlyr = 1
    for i in range(0,ncell):
        for h in range(0, ndays):
            for k in range(0,ignd):
                if maxannualactlyr[i] < zbotw[k,i]:
                    break
                botlyr = k

            for j in range(0, iccp2):
                if botlyr == ignd:
                    unfrozenroot[h,j,:,i] = rmatctem[h,j,:,i]
                else:
                    frznrtlit = np.sum(rmatctem[h,j,botlyr + 0:ignd,i])
                    for k in range(0, botlyr):
                        unfrozenroot[h,j,k,i] = rmatctem[h,j,k,i] + rmatctem[h,j,k,i] / (1.0 - frznrtlit) *frznrtlit

    return unfrozenroot

def rmatctemf(zbotw, alpha, rootmass, avertmas, abar, soildpth, maxannualactlyr, mxrtdpth):
    """
    Function that calculates the variable rmatctem, the fraction of live roots per soil layer per pft
    :param zbotw: Bottom of soil layers [m] dim(ignd)
    :param alpha: Parameter determining how roots grow dim(icc)
    :param rootmass: Root mass [kg C m^-2] dim(ndays,icc)
    :param avertmas: Average root biomass [kg C m^-2] dim(icc)
    :param abar: Parameter determining average root profile dim(icc)
    :param soildpth: Soil permeable depth [m] dim(scalar)
    :param maxannualactlyr: Active layer thickness maximum over the e-folding period specified by parameter eftime [m] dim(scalar)
    :param mxrtdpth: Maximum rooting depth [m] dim(icc)
    :return: rmatctem, fraction of live roots in each soil layer for each pft dim(icc, ignd)
    """

    ndays = rootmass.shape[0]
    # Estimating parameters b and alpha
    b = abar * (avertmas ** alpha)
    useb = b                                        # per pft
    usealpha = alpha                                # per pft
    rootdpth = (4.605 * (rootmass ** alpha)) / b    # Depth of roots [m]
    a = np.zeros([ndays,11])

    # Defining conditions on rootdpth
    i1 = np.where(rootdpth > np.minimum(np.minimum(mxrtdpth, soildpth), maxannualactlyr))
    i2 = np.where(rootdpth <= np.minimum(np.minimum(mxrtdpth, soildpth), maxannualactlyr))

    # Applying conditions
    rootdpth[i1] = np.minimum(np.minimum(mxrtdpth[i1[1]], soildpth), maxannualactlyr)
    a[i1] = 4.605 / rootdpth[i1]
    a[i2] = useb[i2[1]] / (rootmass[i2] ** usealpha[i2[1]])

    # If rootdpth = 0 then 100% of roots are on top layer
    i3 = np.where(rootdpth <= 1e-12)
    a[i3] = 100.0

    # Computing rmatctemp
    kend = np.zeros([ndays,11]) + 9999                      # soil layer in which the roots end, initialized at a dummy value
    totala = 1.0 - np.exp(- a * rootdpth)
    dzbotw = np.tile(zbotw, (11,1,ndays)).transpose(2,0,1)   # Modifying shape for operations on array
    dzroot = np.tile(rootdpth,(20,1,1)).transpose(1,2,0)     # Modifying shape for operations on array

    # Finding in which soil layer the roots end
    ishallow = np.where(rootdpth <= zbotw[0])          # Roots end before or in first soil layer
    iinbetween = np.where((dzroot <= dzbotw) & (dzroot > np.roll(dzbotw, 1, axis=2)))  # roots are at this soil layer
    kend[ishallow] = 0                              # Layer 0
    kend[iinbetween[0],iinbetween[1]] = iinbetween[2]             # Layer at which the roots end (per pft)

    # Computing rmatctem
    ipft = np.arange(0,len(kend),1)
    i = np.where(kend < 1e6)
    i = (i[0],i[1],kend.flatten().astype(int))
    ii = kend.flatten()
    # applying conditions
    etmp = np.exp(-(a * dzbotw.transpose(2,0,1))).transpose(1,2,0)             # in general
    rmatctema = (np.roll(etmp,1,axis=2) - etmp) / np.tile(totala, (20,1,1)).transpose(1,2,0)  # in general
    etmp[:,:,0] = np.exp(-a*dzbotw[:,:,0])               # etmp at first layer
    rmatctema[:,:,0] = (1 - etmp[:,:,0] / totala)       # rmatctem at first layer

    # Looping on every end layer and computing rmat value there
    for k in range(0,kend.shape[0]):
        for h in range(0,kend.shape[1]):
            etmp[k,h,kend[k,h].astype(int)] = -np.exp(-a[k,h]*rootdpth[k,h])
            rmatctema[k,h,kend[k,h].astype(int)] = (etmp[k,h,kend[k,h].astype(int) - 1] - etmp[k,h,kend[k,h].astype(int)]) / totala[k,h]

            rmatctema[k,h,kend[k,h].astype(int)+1:] = 0 # Setting rmatctema to zeros when deeper than kend

    rmatctema[ishallow[0],ishallow[1],0] = 1                      # for pft where roots stop at first layer, 100% in first
    rmatctema[ishallow[0],ishallow[1],1:] = 0                     # 0% in all other layers

    return rmatctema
@jit
def rmatctemf2(zbotw, alpha, rootmass, avertmas, abar, soildpth, maxannualactlyr, mxrtdpth):
    """
    Take two at coding rmatctem hopefully it works this time
    :param zbotw: Bottom of soil layers [m] dim(ignd)
    :param alpha: Parameter determining how roots grow dim(icc)
    :param rootmass: Root mass [kg C m^-2] dim(ndays,icc)
    :param avertmas: Average root biomass [kg C m^-2] dim(icc)
    :param abar: Parameter determining average root profile dim(icc)
    :param soildpth: Soil permeable depth [m] dim(scalar)
    :param maxannualactlyr: Active layer thickness maximum over the e-folding period specified by parameter eftime [m] dim(scalar)
    :param mxrtdpth: mxrtdpth: Maximum rooting depth [m] dim(icc)
    :return: rmatctem, fraction of live roots in each soil layer for each pft dim(icc, ignd)
    """

    iccp2 = alpha.shape[0]
    ignd = zbotw.shape[0]
    ndays = rootmass.shape[0]
    ncell = zbotw.shape[1]
    abszero = 1e-12
    # estimate parameter b of variable root profile parameterization
    b = abar * (avertmas ** alpha)
    # Estimate 99% rooting depth
    useb = b
    usealpha = alpha

    a = np.zeros((ndays,iccp2,ncell))
    totala = np.zeros((ndays,iccp2,ncell))
    rmatctem = np.zeros((ndays,iccp2, ignd,ncell))
    etmp = np.zeros((ndays, iccp2, ignd,ncell))
    tab = np.zeros((ndays,iccp2,ncell))
    for j in range(0,ncell):
        rootdpth = (4.605 * (rootmass[:,:,j] ** alpha)) / b
        for i in range(0,iccp2):
            for k in range(0,ndays):
                if rootdpth[k,i] > min(soildpth[j],maxannualactlyr[j],zbotw[ignd-1,j],mxrtdpth[i]):
                    rootdpth[k,i] = min(soildpth[j],maxannualactlyr[j],zbotw[ignd-1,j],mxrtdpth[i])
                    if rootdpth[k,i] <= abszero:
                        a[k,i,j] = 100.0
                    else:
                        a[k,i,j] = 4.605 / rootdpth[k,i]
                else:
                    if rootmass[k,i,j] <= abszero:
                        a[k,i,j] = 100.0
                    else:
                        a[k,i,j] = useb[i] / (rootmass[k,i,j] ** usealpha[i])
        for i in range(0,iccp2):
            for k in range(0,ndays):

                kend = 9999 # Initialize at dummy value

                # Using parameter 'a' we can find fraction of roots in each soil layer
                zroot = rootdpth[k,i]
                totala[k,i,j] = 1.0 - np.exp(-a[k,i,j] * zroot)

                # If rootdepth is shallower than the bottom of the first layer
                if zroot <= zbotw[0,j]:
                    rmatctem[k,i,0,j] = 1.0
                    rmatctem[k,i,1:,j] = 0.0
                    kend = 0
                else:
                    for tempk in range(1,ignd):
                        if (zroot <= zbotw[tempk,j]) & (zroot > zbotw[tempk-1,j]):
                            kend = tempk
                    if kend == 9999:
                        print('ERROR KEND IS NOT ASSIGNED')
                    etmp[k,i,0,j] = np.exp(-a[k,i,j] * zbotw[0,j])
                    rmatctem[k,i,0,j] = (1.0 - etmp[k,i,0,j]) / totala[k,i,j]
                    if kend == 1:
                        etmp[k,i,kend,j] = np.exp(-a[k,i,j] * zroot)
                        rmatctem[k,i,kend,j] = (etmp[k,i,kend-1,j] - etmp[k,i,kend,j]) / totala[k,i,j]
                    elif kend > 1:
                        for tempk in range(1,kend):
                            etmp[k,i,tempk,j] = np.exp(-a[k,i,j] * zbotw[tempk,j])
                            rmatctem[k,i,tempk,j] = (etmp[k,i,tempk-1,j] - etmp[k,i,tempk,j]) / totala[k,i,j]

                        etmp[k,i,kend,j] = np.exp(-a[k,i,j] * zroot)
                        rmatctem[k,i,kend,j] = (etmp[k,i,kend-1,j] - etmp[k,i,kend,j]) / totala[k,i,j]

                tab[k,i,j] = kend
    rmatctem = unfrozenroot(maxannualactlyr, zbotw, rmatctem)

    dummy = 1
    return rmatctem

def rmatctemf3(zbotw, alpha, rootmass, avertmas, abar, soildpth, maxannualactlyr, mxrtdpth,ncell):
    """
    Take two at coding rmatctem hopefully it works this time
    :param zbotw: Bottom of soil layers [m] dim(ignd)
    :param alpha: Parameter determining how roots grow dim(icc)
    :param rootmass: Root mass [kg C m^-2] dim(ndays,icc)
    :param avertmas: Average root biomass [kg C m^-2] dim(icc)
    :param abar: Parameter determining average root profile dim(icc)
    :param soildpth: Soil permeable depth [m] dim(scalar)
    :param maxannualactlyr: Active layer thickness maximum over the e-folding period specified by parameter eftime [m] dim(scalar)
    :param mxrtdpth: mxrtdpth: Maximum rooting depth [m] dim(icc)
    :return: rmatctem, fraction of live roots in each soil layer for each pft dim(icc, ignd)
    """

    iccp2 = alpha.shape[0]
    ignd = zbotw.shape[0]
    ndays = rootmass.shape[0]
    abszero = 1e-12
    # estimate parameter b of variable root profile parameterization
    b = abar * (avertmas ** alpha)
    # Estimate 99% rooting depth
    useb = b
    usealpha = alpha

    a = np.zeros([ndays,iccp2,ncell])
    totala = np.zeros([ndays,iccp2,ncell])
    rmatctem = np.zeros([ndays,iccp2, ignd,ncell])
    etmp = np.zeros([ndays, iccp2, ignd,ncell])
    tab = np.zeros([ndays,iccp2,ncell])
    for j in range(0,ncell):
        rootdpth = (4.605 * (rootmass[:,:,j] ** alpha)) / b
        for i in range(0,iccp2):
            for k in range(0,ndays):
                if rootdpth[k,i] > min(soildpth,maxannualactlyr,zbotw[ignd-1],mxrtdpth[i]):
                    rootdpth[k,i] = min(soildpth,maxannualactlyr,zbotw[ignd-1],mxrtdpth[i])
                    if rootdpth[k,i] <= abszero:
                        a[k,i] = 100.0
                    else:
                        a[k,i] = 4.605 / rootdpth[k,i]
                else:
                    if rootmass[k,i] <= abszero:
                        a[k,i] = 100.0
                    else:
                        a[k,i] = useb[i] / (rootmass[k,i] ** usealpha[i])
        for i in range(0,iccp2):
            for k in range(0,ndays):

                kend = 9999 # Initialize at dummy value

                # Using parameter 'a' we can find fraction of roots in each soil layer
                zroot = rootdpth[k,i]
                totala[k,i] = 1.0 - np.exp(-a[k,i] * zroot)

                # If rootdepth is shallower than the bottom of the first layer
                if zroot <= zbotw[0]:
                    rmatctem[k,i,0] = 1.0
                    rmatctem[k,i,1:] = 0.0
                    kend = 0
                else:
                    for tempk in range(1,ignd):
                        if (zroot <= zbotw[tempk]) & (zroot > zbotw[tempk-1]):
                            kend = tempk
                    if kend == 9999:
                        print('ERROR KEND IS NOT ASSIGNED')
                    etmp[k,i,0] = np.exp(-a[k,i] * zbotw[0])
                    rmatctem[k,i,0] = (1.0 - etmp[k,i,0]) / totala[k,i]
                    if kend == 1:
                        etmp[k,i,kend] = np.exp(-a[k,i] * zroot)
                        rmatctem[k,i,kend] = (etmp[k,i,kend-1] - etmp[k,i,kend]) / totala[k,i]
                    elif kend > 1:
                        for tempk in range(1,kend):
                            etmp[k,i,tempk] = np.exp(-a[k,i] * zbotw[tempk])
                            rmatctem[k,i,tempk] = (etmp[k,i,tempk-1] - etmp[k,i,tempk]) / totala[k,i]

                        etmp[k,i,kend] = np.exp(-a[k,i] * zroot)
                        rmatctem[k,i,kend] = (etmp[k,i,kend-1] - etmp[k,i,kend]) / totala[k,i]

                tab[k,i] = kend
    rmatctem = unfrozenroot(maxannualactlyr, zbotw, rmatctem)

    dummy = 1
    return rmatctem