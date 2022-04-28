import numpy as np
import fturbation
import fenv
import time


def model(param,ncell,iccp2,ignd,litrmass,soilcmas,tf,delt,ndays, thice, thpor, thliq, psisat, tbar, b,
          itermax,spinup,zbotw,actlyr,SAND,fcancmx,delzw,root_input,fLeafLitter,fStemLitter,rank):
    """
    Function that simulates
    :param param: PFT specific parameters t obe optimized
    :param litrmass: Mass of carbon in litter pool, per layer, per PFT [kg C m^-2]
    :param soilcmas: Mass of carbon in soil pool, per layer, per PFT [kg C m^-2
    :param ignd: Number of soil layers
    :param iccp2: Number of CTEM PFTs + bare + LUC
    :param itermax: Number of iterations in the simulation [days]
    :param tf: Length of the forcing data [yr]
    :param delt: Length of a timestep [yr]
    :param spinup: Number of spinup iterations
    :param ndays: Number of days of data
    :param thice: Frozen water content [m^3 m^-3]
    :param thpor: Total soil porority [m^3 m^-3]
    :param thliq: Liquid water content [m^3 m^-3]
    :param psisat: Soil moisture suction at saturation [m]
    :param isand: Flag for soil type -4 = ice, -3 = rock
    :param tbar: Temperature at each soil layer [K]
    :param b: Clapp and Hornberger empirical param []
    :param zbotw: Bottom of permeable soil layers [m]
    :param actlyr: Max annual active layer depth [m]
    :param SAND: Sand content of soils [%]
    :param delzw: Thickness of permeable part of soil layers [m]
    :param root_input: Root input to litter pool [kg C m^-2 yr-1]
    :param fLeafLitter: Leaf input to litter pool [kg C m^-2 yr^-1]
    :param fStemLitter: Stem input to litter pool [kg C m^-2 yr^-1]
    :param fcancmx: PFT coverage fraction of gridcells + bare + LUC
    :param ncell: Number of gridcell to simulate
    :return: litter and soil pool size per layer [kg C m^-2]
    """
    # Variables needed for run
    # Array of pft specific variables, 9 CTEM pfts + Bare + LUC

    apfts = np.array([[param['bsratelt_NdlEvgTr'], param['bsratelt_NdlDcdTr'], param['bsratelt_BdlEvgTr'],
                       param['bsratelt_BdlDCoTr'], param['bsratelt_BdlDDrTr'], param['bsratelt_CropC3'],
                       param['bsratelt_CropC4'], param['bsratelt_GrassC3'], param['bsratelt_GrassC4'],
                       param['bsratelt_Bare'], param['bsratelt_LUC']],
         [param['bsratesc_NdlEvgTr'], param['bsratesc_NdlDcdTr'], param['bsratesc_BdlEvgTr'],
          param['bsratesc_BdlDCoTr'], param['bsratesc_BdlDDrTr'], param['bsratesc_CropC3'], param['bsratesc_CropC4'],
          param['bsratesc_GrassC3'], param['bsratesc_GrassC4'], param['bsratesc_Bare'], param['bsratesc_LUC']],
         [param['humicfac_NdlEvgTr'], param['humicfac_NdlDcdTr'], param['humicfac_BdlEvgTr'],
          param['humicfac_BdlDCoTr'], param['humicfac_BdlDDrTr'], param['humicfac_CropC3'], param['humicfac_CropC4'],
          param['humicfac_GrassC3'], param['humicfac_GrassC4'], param['humicfac_Bare'], param['humicfac_LUC']]])
    threshold = 100  # TODO decide of a acceptable threshold

    # Initializing soil C content vector
    litrplsz = litrmass[:,:iccp2,:].transpose(1,0,2)   # initial carbon mass of litter pool [kg C / m^2]
    soilplsz = soilcmas[:, :iccp2,:].transpose(1,0,2)  # initial carbon mass of soil pool [kg C / m^2]

    # Initializing litter input array
    Cinput = np.zeros([iccp2, ignd, ncell])  # Pool carbon input

    # arrays for outputs
    litter = np.empty([ignd, ncell, 20440])
    soil = np.zeros([ignd, ncell, 20440])
    resp = np.zeros([ncell, 20440])

    count = 0  # iteration count
    output_count = 0  # number of output counts
    eqflag = 0  # Equilibrium flag, raised once equilibrium is reached

    # Assigning parameter values from algorithm to their corresponding variable
    bsratelt = apfts[0, :]  # turnover rate of litter pool [kg C/kg C.yr]
    bsratesc = apfts[1, :]  # turnover rate of soil pool [kg C/kg C.yr]
    humicfac = apfts[2, :]  # Humification factor
    kterm = param['kterm']  # Constant that determines the depth at which turbation stops
    biodiffus = param['biodiffus'] # Diffusivity coefficient for bioturbation [m^2/day]
    cryodiffus = param['cryodiffus'] # Diffusivity coefficient for cryoturbation [m^2/day]
    tanha = param['tanha'] # Constant a for tanh formulation of respiration Q10 determination
    tanhb = param['tanhb'] # Constant b for tanh formulation of respiration Q10 determination
    tanhc = param['tanhc'] # Constant c for tanh formulation of respiration Q10 determination
    tanhd = param['tanhd'] # Constant d for tanh formulation of respiration Q10 determination
    r_depthredu = param['r_depthredu']
    tcrit = param['t_crit']
    frozered = param['frozered']
    mois_a = param['moisa']
    mois_b = param['moisb']
    mois_c = param['moisc']

    reduceatdepth = np.exp(-delzw / r_depthredu)

    # Computing environmental modifiers for the whole run
    envmodltr, envmodsol = fenv.fenv_numba(ndays, thice, thpor, thliq, psisat, ignd, SAND, tbar, b, ncell, tanha, tanhb,
                                     tanhc, tanhd, tcrit, frozered, mois_a, mois_b, mois_c)

    eqarr = np.zeros(5)  # Mobile array for equilibrium check
    sample = int(61 / delt)
    # Time loop
    #print('Starting model run')
    while count < itermax:
        if not eqflag:
            t = count % sample  # cycling on the length of imput files
        # Computing environmental modifier at current timestep
        envltr = envmodltr[t]  # Litter pool environmental modifier at current timestep
        envsol = envmodsol[t]  # Soil pool environmental modifier at current timestep

        # Modifying transfert coeff if spinup=True
        if count <= spinup:
            spinfast = 10  # transfer coeff from litter pool to soil
        else:
            spinfast = 1

        # Computing input vector at current time step
        Cinput = np.zeros([iccp2, ignd, ncell])
        Cinput[:, 0, :] = fLeafLitter[t] + fStemLitter[t]  # Carbon input to litter pool at timestep t [Kg C m^-2 yr^-1]
        Cinput += root_input[t] # Carbon input from roots at timestep t [kg C m^-2 yr^-1]

        # Computing respiration over time step for both pool
        ltresveg = bsratelt * (litrplsz * envltr * reduceatdepth).transpose(1,2,0) # [kg C m^-2 yr^-1]
        scresveg = bsratesc * (soilplsz * envsol * reduceatdepth).transpose(1,2,0)  # [kg C m^-2 yr^-1]

        # Computing pool size change in both pool
        dx1 = Cinput - (ltresveg * (1 + humicfac)).transpose(2, 0, 1)
        dx2 = spinfast * ((ltresveg * humicfac) - scresveg).transpose(2, 0, 1)

        # Updating pools
        litrplsz += dx1 * delt  # Updating litter pool [kg C m^-2]
        soilplsz += dx2 * delt  # Updating soil pool [kg C m^-2]

        # Calling turbation subroutine
        litrplsz,soilplsz = fturbation.turbation(litrplsz,soilplsz,zbotw,ignd,kterm,actlyr[t,:],cryodiffus,biodiffus,
                                                 spinfast,SAND,ncell,delt,iccp2)

        count += 1  # Iteration count
        #print(count)
        # Equilibrium check
        if count % 20000 == 0:  # Check for equilibrium every 10000 iterations
            print(count, ' / ', itermax, 'on rank: ',rank)
            totsol = np.nansum(soilplsz)  # Summing on all gridcells for soil pool
            eqarr = np.append(eqarr, totsol)  # Adding new value ton equilibrium array
            eqarr = np.delete(eqarr, 0)  # Removing oldest value in the array
            x = np.linspace(0, len(eqarr), len(eqarr))  # x-coords for slope calculations
            slope = np.polyfit(x, eqarr, 1)[0]  # Computing slope of equilibrium array
            #print(slope)
            if np.abs(slope) <= threshold:  # If threshold is crossed, raise equilibrium flag
                eqflag = 1


        if eqflag:  # Once equilibrium flag is up, start outputting
            #print('EQ FLAG RAISED')
            #print(count)
            gridltr = np.sum(litrplsz.transpose(1, 0, 2) * fcancmx, axis=1)  # Litter pool size at grid level
            gridsol = np.sum(soilplsz.transpose(1, 0, 2) * fcancmx, axis=1)  # SoilC pool size at grid level
            gridrsp = np.sum(((ltresveg + scresveg) * fcancmx.transpose()),axis=(0,2)) # Respiration at grid level
            t = output_count + 22265
            # Adding to output arrays
            litter[:, :, output_count] = gridltr
            soil[:, :, output_count] = gridsol
            resp[:,output_count] = gridrsp
            output_count += 1  # Updating number of outputs

        if output_count >= 20440:  # Once equilibrium is reached we output tf * 365 more iterations and stop the simul
            break
    return litter, soil, resp
