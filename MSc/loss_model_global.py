import numpy as np
import xarray as xr
import func_model
import pickle
from scipy.interpolate import interp1d
import taylor_score_f as ts
import frmatctem
import time
from datetime import date
from numba import jit


def wosiscomp(wosis, classic, classic_zbot):
    """
    Function that computes the taylor score from the comparison between CLASSIC pool size and WOSIS data.
    :param wosis: Wosis soil carbon data for each gridcell [kg C m^-2]
    :param classic: Model pool size output for eah gridcell [kg C m^-2]
    :param classic_zbot: Bottom of permeable soil layers [m]
    :return: Taylor score from comparison between CLASSIC and WOSIS
    """
    # Empty arrays to store formatted data
    av_tab = np.array([])       # Array of average value from Wosis at a certain depth
    std_tab = np.array([])      # Array of std from Wosis at a certain depth
    model_tab = np.array([])    # Model value

    for h in range(0,len(wosis)):           # Looping on gridcell containing Wosis data
        # Observation data
        depth = np.array(wosis[h][1][0])    # List of all layer depths in the gridcell
        orgc = np.array(wosis[h][0][0])     # Array of wosis soil C value for all layers in the gridcell

        # Model output
        classic_orgc = np.mean(classic, axis=2)[:,h]  # Model simulated soil C
        temp = np.zeros(21)                 # Formatting to add layer at surface with 0 soil C
        temp[1:] = classic_orgc
        classic_orgc = temp
        values = np.unique(depth)           # All possible values of depth in the gridcell
        score = 0  # Initial score

        for k in range(0, len(values)):     # Looping on all possible depth value in the grid cell
            current_depth = values[k]       # Current depth that is being used
            ind = np.where(depth == values[k])  # Index of all layers that are at the current depth

            av = np.mean(orgc[ind])         # Average value of the layers at current depth
            std = np.std(orgc[ind])         # std value of the layers at current depth
            f = interp1d(classic_zbot, classic_orgc) # Interpolation function from model's layer depth and soil C values
            classic_inter_orgc = f(current_depth)    # Interpolating CLASSIC soil C at current depth

            # Adding to empty arrays
            av_tab = np.append(av_tab,av)
            std_tab = np.append(std_tab,std)
            model_tab = np.append(model_tab, classic_inter_orgc)

    # Removing mysterious NaNs
    #inan = np.where((~np.isnan(av_tab)) & (~np.isnan(std_tab)))
    #av_tab = av_tab[inan]
    #std_tab = std_tab[inan]
    #model_tab = model_tab[inan]

    # Computing taylor score
    wosis_score,wosis_coeff = ts.score(av_tab,model_tab,1)

    return wosis_score


def srdbcomp(srdb, classic):
    """
    Function that computes taylor score between CLASSIC respiration output and SRDB respiration data
    :param srdb: Soil Respiration DataBase data
    :param classic: SImulated soil respiration
    :return: Taylor score of comparison between
    """
    srdb_resp = srdb[:,2]       # Soil respiration data from SRDB
    srdb_resp_std = srdb[:,3]   # Std of soil respiration data from SRDB
    srdb_begin_yr = srdb[:,4]   # Start year of observation data
    srdb_stop_yr = srdb[:,5]    # Stop year

    # Creating empty arrays to store data for comparison
    classic_resp = np.array([]) # Simulated soil respiration
    classic_std = np.array([])  # Std of simulated soil respiration
    observd_resp = np.array([]) # Observational soil respiration data
    observd_std = np.array([])  # Std of observational soil resp data

    # Getting every datapoint for comparison
    for i in range(0,len(srdb)): # Looping on each gridcell containing SRDB data
        # Converting start/stop year to index of the model output
        ibegin = (np.floor((365*srdb_begin_yr[i][0][:,0] - 1900*365 - 22265 + 1)).astype(int)) # array idex for begin yr
        istop = (np.floor(365*srdb_stop_yr[i][0][:,0] - 1900*365 - 22265 + 1)).astype(int)     # array index for stop yr

        for j in range(0,len(ibegin)): # Looping on all data in a single gridcell
            # Adding datapoints to empty arrays
            classic_resp = np.append(classic_resp,np.mean(classic[i,ibegin[j]:istop[j]]))
            classic_std = np.append(classic_std, np.std(classic[i,ibegin[j]:istop[j]]))
            observd_resp = np.append(observd_resp, srdb_resp[i][0][j,0])
            observd_std = np.append(observd_std,srdb_resp_std[i][0][j,0])

    # Computing Taylor score
    srdb_score, srdb_coeff = ts.score(observd_resp,classic_resp,1)


    return srdb_score


def loss(params):
    # Loading inupt files and formatting to number of gridcells desired
    init = xr.open_dataset('/home/charlesgauthier/project/global_opt_files/rsFile_modified.nc') # Restart file

    ncell = 10#envmodltr.shape[2]  # Number of gridcells

    litrmass = np.load('/home/charlesgauthier/project/test_files/litrmass_test.npy')       #TODO !!!TEST VALUE!!!
    litrmass = litrmass[:,:,:ncell]

    soilcmass = np.load('/home/charlesgauthier/project/test_files/soilcmas_test.npy')      #TODO !!!TEST VALUE!!!
    soilcmass = soilcmass[:,:,:ncell]

    fLeafLitter = np.load('/home/charlesgauthier/project/test_files/fVegLitter_test.npy')   #TODO !!!TEST VALUE!!!
    fLeafLitter = fLeafLitter[:,:,:ncell]

    fStemLitter = np.load('/home/charlesgauthier/project/test_files/fVegLitter_test.npy') # TODO !!!TEST VALUE!!!
    fStemLitter = fStemLitter[:,:,:ncell]

    fRootLitter = np.load('/home/charlesgauthier/project/test_files/fVegLitter_test.npy') # TODO !!!TEST VALUE!!!
    fRootLitter = fRootLitter[:,:,:ncell]


    actlyr = np.load('/home/charlesgauthier/project/test_files/actlyr_test.npy')           #TODO !!!TEST VALUE!!!
    actlyr = actlyr[:,:ncell]

    fcancmx = np.load('/home/charlesgauthier/project/test_files/fcancmx_test.npy')  # TODO !!!TEST VALUE!!!
    fcancmx = fcancmx[:, :ncell]

    SAND = np.load('/home/charlesgauthier/project/test_files/SAND_test.npy')  # TODO !!!TEST VALUE!!!
    SAND = SAND[:, :ncell]

    CLAY = np.load('/home/charlesgauthier/project/test_files/CLAY_test.npy') # TODO !!!TEST VALUE!!!
    CLAY = CLAY[:, :ncell]

    isand = np.where((SAND == -4) | (SAND == -3))       # Flag ,-4 = ice, -3 = ice

    zbotw = np.load('/home/charlesgauthier/project/test_files/zbotw_test.npy')  # TODO !!!TEST VALUE!!!
    zbotw = zbotw[:, :ncell]

    delzw = np.load('/home/charlesgauthier/project/test_files/delzw_test.npy') # TODO !!! TEST VALUE!!!
    delzw = delzw[:, :ncell]

    tbar = np.load('/home/charlesgauthier/project/test_files/tbar_test.npy') # TODO !!! TEST VALUE!!!
    tbar = tbar[:, :, :ncell]

    mliq = np.load('/home/charlesgauthier/project/test_files/mliq_test.npy') # TODO !!!TEST VALUE!!!
    mliq = mliq[:,:,:ncell]
    thliq = mliq/1000/delzw # Conversion to m^3/m^3

    mice = np.load('/home/charlesgauthier/project/test_files/mice_test.npy') # TODO !!!TEST VALUE!!!
    mice = mice[:,:,:ncell]
    thice = mice/1000/delzw # Conversion to m^3/m^3

    thpor = (-0.126 * SAND + 48.9) / 100  # Soil total porority [m^3 m^-3]
    thpor[isand] = 0  # No value where soil is rock or ice
    b = 0.159 * CLAY + 2.91  # Clapp and Hornberger empirical param []
    b[isand] = 0  # No value where soil is rock or ice
    psisat = 0.01 * np.exp(-0.0302 * SAND + 4.33)  # Soil moisture suction at saturation [m]
    psisat[isand] = 0

    # Global variables
    ignd = init.dims['layer']
    iccp2 = init.dims['iccp2']
    ndays = tbar.shape[0]
    tf = int((ndays)/365)

    itermax = 180000
    delt = 1/365
    spinup = 10000

    # pft specific params
    alpha = np.array([0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0,0])
    avertmas = np.array([1.85, 1.45, 2.45, 2.10, 2.10,0.1, 0.1, 0.7,0.7,0,0])
    abar = np.array([4.70,5.86,3.87,3.46,3.97,3.97,3.97,5.86,4.92,4,4])
    mxrtdpth = np.array([3,3,5,5,3,2,2,1,1,0,0])

    # Computing rmatctem
    rmatctem = np.load('/home/charlesgauthier/project/rmatctem_test.npy') #TODO PRE PROCESS BEFORE RUN !!!TEST VALUE!!!
    rmatctem = rmatctem[:,:,:,:ncell]
    root_input = (fRootLitter * rmatctem.transpose(2,0,1,3)).transpose(1,2,0,3)

    # Simulating soilC
    arg1, arg2, arg3 = func_model.model(params,ncell,iccp2,ignd,litrmass,soilcmass,tf,delt,ndays, thice, thpor, thliq,
                                        psisat, tbar, b, itermax,spinup,zbotw,actlyr,SAND,fcancmx,delzw,
                                        root_input,fLeafLitter,fStemLitter,1)

    # Reading in formated woSIS/srdb dataset
    with open('/home/charlesgauthier/project/wosis_opt/wosis_srdb_grid.txt', 'rb') as fp:
        grid = pickle.load(fp)
    grid = grid[:ncell]

    # Reading in dataset_flag array
    dataset_flag = np.load('/home/charlesgauthier/project/wosis_opt/dataset_flag.npy')
    dataset_flag = dataset_flag[:ncell]

    # Depth of soil layers
    delz = init['DELZ'].data[:]                         # Ground layer thickness [m]
    zbot = np.zeros_like(delz)                          # Dept of the bottom of each soil layer [m]
    for i in range(0, len(delz)):
        zbot[i] = np.sum(delz[:i+1])
    classic_zbot = np.zeros(len(zbot)+1)
    classic_zbot[1:] = zbot

    # Wosis score
    iwosis = np.where((dataset_flag ==1) | (dataset_flag==2))
    wosis_score = wosiscomp(np.array(grid, dtype=object)[iwosis], arg1[:,iwosis,:][:,0,:,:],classic_zbot)

    # Srdb score
    isrdb = np.where((dataset_flag == 2) | (dataset_flag == 3))
    srdb_score = srdbcomp(np.array(grid, dtype=object)[isrdb], arg3[isrdb])

    # Computing score
    alpha = 0.7
    score = alpha*wosis_score + (1-alpha)*srdb_score

    return 1/score
