import numpy as np
from numba import jit
from numba import njit

@jit
def tridiag(a, b, c, r, u):
    """
    Tridiagonal algorithm matrix solver, see numerical recipes in Fortran 90, equation 2.4.1
    :param a:
    :param b:
    :param c:
    :param r:
    :param u:
    :return:
    """
    gam = np.zeros(len(b))

    n = len(b)

    bet = b[0]
    u[0] = r[0] / bet

    # Decomposition and foward substitution
    for i in range(1, n):
        gam[i] = c[i-1] / bet

        bet = b[i] - a[i] * gam[i]

        u[i] = (r[i] - a[i] * u[i-1]) / bet

    # Back substitution
    for j in range(n-2,0,-1):
        u[j] = u[j] - gam[j+1] * u[j+1]
    return u

@jit
def turbation(litrmass, soilcmass, zbotw, ignd,kterm, actlyr,cryodiffus,biodiffus,spinfast,SAND,ncell,delt,iccp2):
    """
    Function that performs the turbation scheme on litter and soil pools using the Crank-Nicolson method, described
    here: https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method See soilCProcesses.f90 subroutine for source.
    Compatible with numba, coded in a fortran fashion
    :param litrmass: Litter pool mass [kg C m^-2] dim(iccp2,ignd,ncell)
    :param soilcmass: Soil pool mass [kg C m^-2] dim(iccp2,ignd,ncell)
    :param zbotw: Bottom of permeable soil layer [m] dim(ignd, ncell)
    :param ignd: Number of soil layers
    :param kterm: Constant used in determination of depth at which cryoturbation ceases
    :param actlyr: Active layer depth at current timestep [m] dim(ncell)
    :param cryodiffus: Cryoturbation diffusion coefficient [m^2 day^-1]
    :param biodiffus: Bioturbation diffusion coefficient [m^2 day^-1]
    :param spinfast: Spin up constant
    :param SAND: Sand content of soils [%] dim(ignd, ncell)
    :param ncell: Number of gridcells
    :param delt: Length of timestep [yr]
    :param iccp2: Number of pfts
    :return: litrmass, soilcmass : Updated pool mass
    """

    for i in range(0,ncell):
        # First we find the bottom of the permeable soil column by looking at the isand flags
        botlyr = 0
        for j in range(0,ignd):
            if (SAND[j,i] == -3) or (SAND[j,i] == -4):
                break
            botlyr = j
        # Next if the gridcell has spme permeable soil, then botlyr is bottom soil and
        if botlyr > 0:
            botlyr += 2 # We add two more for the boundary layers

            avect = np.zeros(botlyr+1)
            bvect = np.zeros(botlyr+1)
            cvect = np.zeros(botlyr+1)
            rvect_sc = np.zeros(botlyr+1)
            rvect_lt = np.zeros(botlyr+1)
            soilcinter = np.zeros(botlyr+1)
            littinter = np.zeros(botlyr+1)
            depthinter = np.zeros(botlyr+1)
            effectiveD = np.zeros(botlyr+1)

            for k in range(0,iccp2):

                # At the start, store the size of current pool for later balance check
                psoilc = np.sum(soilcmass[k,:,i])
                plit = np.sum(litrmass[k,:,i])

                if psoilc > 1e-12: # Turbation only occurs for PFTs with some soil C
                    # Set up of the tridiagonal solver for soil and litter C.

                    # Boundary condition at surface = 0
                    soilcinter[0] = 0

                    littinter[0]  = 0
                    depthinter[0] = 0

                    # Putting soil/litter C in tridiag arrays
                    soilcinter[1:botlyr] = soilcmass[k,0:botlyr-1,i]
                    littinter[1:botlyr] = litrmass[k,0:botlyr-1,i]
                    depthinter[1:botlyr] = zbotw[0:botlyr-1,i]

                    # Boundary condition at bottom of soil column
                    soilcinter[botlyr] = 0
                    littinter[botlyr] = 0
                    if i ==7:
                        dummy =1
                    # Check for special case where soil is permeable all the way to the bottom
                    if botlyr <= ignd:
                        botthick = zbotw[botlyr-1,i]
                    else:
                        botthick = zbotw[ignd-1,i]

                    depthinter[botlyr] = botthick

                    # Diffusion coefficient computation for each soil layer
                    kactlyr = actlyr[i] * kterm

                    if actlyr[i] <= 1:
                        diffus = cryodiffus  # actlyr is shallow, cryoturbation is dominant
                    else:
                        diffus = biodiffus # Else bioturbation is dominant

                    for l in range(0, botlyr+1):
                        if depthinter[l] < actlyr[i]: # Shallow, so vigorous cryoturb
                            effectiveD[l] = diffus * spinfast

                        elif (depthinter[l] > actlyr[i]) and (depthinter[l] < kactlyr):  # Linear reduction in diff coef
                            effectiveD[l] = diffus * (1 - (depthinter[l] - actlyr[i])/(kterm - 1) * actlyr[i]) * spinfast

                        else:  # Too deep, no cryoturbation assumed
                            effectiveD[l] = 0

                    # Setup of the coefficient for the tridiag matrix algorithm

                    # Upper boundary condition
                    avect[0] = 0.0
                    bvect[0] = 1.0
                    cvect[0] = 0.0
                    rvect_sc[0] = 0.0
                    rvect_lt[0] = 0.0

                    # Loop on soil layers
                    for f in range(1,botlyr):
                        dzm = depthinter[f] - depthinter[f-1]
                        termr = effectiveD[f] * delt / dzm**2
                        avect[f] = -termr
                        bvect[f] = 2 * (1 + termr)
                        cvect[f] = -termr
                        rvect_sc[f] = termr * soilcinter[f-1] + 2 * (1-termr) * soilcinter[f] + termr * soilcinter[f+1]
                        rvect_lt[f] = termr * littinter[f-1] + 2 * (1-termr) * littinter[f] + termr * littinter[f+1]

                    # Bottom boundary condition
                    avect[botlyr] = 0.0
                    bvect[botlyr] = 1.0
                    cvect[botlyr] = 0.0
                    rvect_sc[botlyr] = 0.0
                    rvect_lt[botlyr] = 0.0

                    # Call tridiagonal solver for soil and litter pool
                    soilcinter = tridiag(avect, bvect, cvect, rvect_sc, soilcinter)
                    littinter = tridiag(avect, bvect, cvect, rvect_lt,littinter)

                    # Add updated C back into their arrays
                    soilcmass[k,0:botlyr-1,i] = soilcinter[1:botlyr]
                    litrmass[k,0:botlyr-1,i] = littinter[1:botlyr]

                    # Carbon balance conservation check
                    asoilc = np.sum(soilcmass[k,:,i]) # Ammount after turbation
                    alit = np.sum(litrmass[k,:,i])

                    # Balance for soil C
                    amount_sc = psoilc - asoilc
                    soilcmass[k,0:botlyr-1,i] += amount_sc / (botlyr-1)
                    # Balance for litter C
                    amount_lt = plit - alit
                    litrmass[k,0:botlyr-1,i] += amount_lt / (botlyr-1)

    return litrmass, soilcmass
@jit
def turbation2(litrmass, soilcmass, zbotw, ignd,kterm, actlyr,cryodiffus,biodiffus,spinfast,SAND,ncell,delt,iccp2):
    """
    Function that performs the turbation scheme on litter and soil pools using the Crank-Nicolson method, described
    here: https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method See soilCProcesses.f90 subroutine for source.
    !! Used for sensitivity analysis, single site, multiple params!!!
    :param litrmass: Litter pool mass [kg C m^-2] dim(iccp2,ignd,ncell)
    :param soilcmass: Soil pool mass [kg C m^-2] dim(iccp2,ignd,ncell)
    :param zbotw: Bottom of permeable soil layer [m] dim(ignd, ncell)
    :param ignd: Number of soil layers
    :param kterm: Constant used in determination of depth at which cryoturbation ceases
    :param actlyr: Active layer depth at current timestep [m] dim(ncell)
    :param cryodiffus: Cryoturbation diffusion coefficient [m^2 day^-1]
    :param biodiffus: Bioturbation diffusion coefficient [m^2 day^-1]
    :param spinfast: Spin up constant
    :param SAND: Sand content of soils [%] dim(ignd, ncell)
    :param ncell: Number of gridcells
    :param delt: Length of timestep [yr]
    :param iccp2: Number of pfts
    :return: litrmass, soilcmass : Updated pool mass
    """

    for i in range(0,ncell):
        # First we find the bottom of the permeable soil column by looking at the isand flags
        botlyr = 0
        for j in range(0,ignd):
            if (SAND[j,0] == -3) or (SAND[j,0] == -4):
                break
            botlyr = j
        # Next if the gridcell has spme permeable soil, then botlyr is bottom soil and
        if botlyr > 0:
            botlyr += 2 # We add two more for the boundary layers

            avect = np.zeros(botlyr+1)
            bvect = np.zeros(botlyr+1)
            cvect = np.zeros(botlyr+1)
            rvect_sc = np.zeros(botlyr+1)
            rvect_lt = np.zeros(botlyr+1)
            soilcinter = np.zeros(botlyr+1)
            littinter = np.zeros(botlyr+1)
            depthinter = np.zeros(botlyr+1)
            effectiveD = np.zeros(botlyr+1)

            for k in range(0,iccp2):

                # At the start, store the size of current pool for later balance check
                psoilc = np.sum(soilcmass[k,:,i])
                plit = np.sum(litrmass[k,:,i])

                if psoilc > 1e-12: # Turbation only occurs for PFTs with some soil C
                    # Set up of the tridiagonal solver for soil and litter C.

                    # Boundary condition at surface = 0
                    soilcinter[0] = 0

                    littinter[0] = 0
                    depthinter[0] = 0

                    # Putting soil/litter C in tridiag arrays
                    soilcinter[1:botlyr] = soilcmass[k,0:botlyr-1,i]
                    littinter[1:botlyr] = litrmass[k,0:botlyr-1,i]
                    depthinter[1:botlyr] = zbotw[0:botlyr-1,0]

                    # Boundary condition at bottom of soil column
                    soilcinter[botlyr] = 0
                    littinter[botlyr] = 0

                    # Check for special case where soil is permeable all the way to the bottom
                    if botlyr <= ignd:
                        botthick = zbotw[botlyr-1,0]
                    else:
                        botthick = zbotw[ignd-1,0]

                    depthinter[botlyr] = botthick

                    # Diffusion coefficient computation for each soil layer
                    kactlyr = actlyr * kterm[i]

                    if actlyr <= 1:
                        diffus = cryodiffus[i]  # actlyr is shallow, cryoturbation is dominant
                    else:
                        diffus = biodiffus[i] # Else bioturbation is dominant

                    for l in range(0, botlyr+1):
                        if depthinter[l] < actlyr: # Shallow, so vigorous cryoturb
                            effectiveD[l] = diffus * spinfast

                        elif (depthinter[l] > actlyr) and (depthinter[l] < kactlyr):  # Linear reduction in diff coef
                            effectiveD[l] = diffus * (1 - (depthinter[l] - actlyr)/(kterm[i] - 1) * actlyr) * spinfast

                        else: # Too deep, no cryoturbation assumed
                            effectiveD[l] = 0

                    # Setup of the coefficient for the tridiag matrix algorithm

                    # Upper boundary condition
                    avect[0] = 0.0
                    bvect[0] = 1.0
                    cvect[0] = 0.0
                    rvect_sc[0] = 0.0
                    rvect_lt[0] = 0.0

                    # Loop on soil layers
                    for f in range(1,botlyr):
                        dzm = depthinter[f] - depthinter[f-1]
                        termr = effectiveD[f] * delt / dzm**2
                        avect[f] = -termr
                        bvect[f] = 2 * (1 + termr)
                        cvect[f] = -termr
                        rvect_sc[f] = termr * soilcinter[f-1] + 2 * (1-termr) * soilcinter[f] + termr * soilcinter[f+1]
                        rvect_lt[f] = termr * littinter[f-1] + 2 * (1-termr) * littinter[f] + termr * littinter[f+1]

                    # Bottom boundary condition
                    avect[botlyr] = 0.0
                    bvect[botlyr] = 1.0
                    cvect[botlyr] = 0.0
                    rvect_sc[botlyr] = 0.0
                    rvect_lt[botlyr] = 0.0

                    # Call tridiagonal solver for soil and litter pool
                    soilcinter = tridiag(avect, bvect, cvect, rvect_sc, soilcinter)
                    littinter = tridiag(avect, bvect, cvect, rvect_lt,littinter)

                    # Add updated C back into their arrays
                    soilcmass[k,0:botlyr-1,i] = soilcinter[1:botlyr]
                    litrmass[k,0:botlyr-1,i] = littinter[1:botlyr]

                    # Carbon balance conservation check
                    asoilc = np.sum(soilcmass[k,:,i]) # Ammount after turbation
                    alit = np.sum(litrmass[k,:,i])

                    # Balance for soil C
                    amount_sc = psoilc - asoilc
                    soilcmass[k,0:botlyr-1,i] += amount_sc / (botlyr-1)
                    # Balance for litter C
                    amount_lt = plit - alit
                    litrmass[k,0:botlyr-1,i] += amount_lt / (botlyr-1)

    return litrmass, soilcmass