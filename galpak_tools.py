import numpy as np
import matplotlib.pyplot as plt
import galpak

import os

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import WMAP9 
from scipy.stats import binned_statistic
import matplotlib.backends.backend_pdf

import mpdaf
from mpdaf.obj import Cube
from mpdaf.drs import PixTable
from galpak import MoffatPointSpreadFunction, GaussianLineSpreadFunction
from mpdaf.sdetect import Source

import glob

from astropy.io import fits 

import CAMEL.camel as cml
import CAMEL.create_config as cc


#---------------------------------------------------------------
#---------------------------------------------------------------
# GLOBAL VARIABLES:

L_oii_1 = 3727.10
L_oii_2 = 3729.86

zmin_oii = 4700 / L_oii_1 - 1
zmax_oii = 9300 / L_oii_1 - 1

#----------------------------------------------------------------
#-------------------------------------------------
"""

def get_line_feature(hdul, line="OII3726", feature="SNR"):
    #get a specific line feature from a fits file. For example the SNR of the OII line
    marz_id = get_marz_solution(hdul)
    if marz_id == 0:
        tab_name = "TAB_PLFEL_LINES"
        mz_lines = hdul[tab_name].data
        line_data = mz_lines[mz_lines["LINE"] == line]
        print(line_data[feature])
        if len(line_data[feature]) == 1:
            return float(line_data[feature])

    elif marz_id == 6:
        tab_name = "TAB_PL_LINES"
        mz_lines = hdul[tab_name].data
        line_data = mz_lines[mz_lines["LINE"] == line]
        print(line_data[feature])
        if len(line_data[feature]) == 1:
            return float(line_data[feature])

    elif marz_id != -1:
        tab_name = "TAB_MZ" + str(marz_id) + "_LINES"
        mz_lines = hdul[tab_name].data
        line_data = mz_lines[mz_lines["LINE"] == line]
        print(line_data[feature])
        if len(line_data[feature]) == 1:
            return float(line_data[feature])

    else:
        return 10000
"""
#----------------------------------------------------------------------------
def get_line_feature(src, line="OII3726", feature="SNR"):
    """
    get a specific line feature from a fits file. For example the SNR of the OII line
    """
    PL_LINES = src.tables["PL_LINES"]

    if line in PL_LINES["LINE"]:
        l = PL_LINES[PL_LINES["LINE"] == line]
        f1 = l["FAMILY"] == "all"
        f2 = l["FAMILY"] == "balmer"
        l = l[f1 | f2]
        return l[feature][0]

    return 0

#------------------------------------------------------------------------------

def get_z_src(src):
    """
    get the redshift from a source file.
    """

    try:
        z_hdu = hdul["Z"].data
        z_src = z_hdu["Z"][0]
    except:
        print("NO REDSHFIT AVAILABLE")
        z_src = -1.0
    return z_src


#------------------------------------------------------------------------

def substract_continuum(src_path, output_path, snr_min = 15, line="OII"):
    """
    Take a source file as an input, substract the continuum of the cube around a given line/doublet
    and save the output.
    inputs:
        -src_path: the path of the source file
        
        -output_path: the path of the galpak directory (not the specific path where the cube must be saved)
        
        -snr_min: the SNR limit of the specified line for the continuum substraction to be performed
        
        - line: the target emission line around which we want to extract the cube. Could be:
                OII, OIII5007, HALPHA, 
        
    output: 
        - a cube of 32 spaxel centered on the emission line.
        - a pdf with the images before and after substraction and the spectrum
    """

    # First we open the source:
    src = Source.from_file(src_path)

    src_name = src_path[-28:-5] # the source name in the format "JxxxxXxxxx_source_xxxxx"
    field =  src_path[-28:-18] # the field id in the format JxxxxXxxxx"

    output_path_field = output_path + field + "/"
    src_output_path = output_path_field + src_name+ "/"
    cube_output_path = src_output_path + src_name + "_cube.fits"

    # We extract the redshift of the source:
    try:
        z_src = src.z["Z"][0]
    except:
        z_src = 0
        print("WARNING: No redshift")

    # we get the SNR:
    if line == "OII":
        try:
            oii_3726_snr = get_line_feature(src, line="OII3726", feature="SNR")
            oii_3729_snr = get_line_feature(src, line="OII3729", feature="SNR")
            line_snr = max(oii_3726_snr, oii_3729_snr)
            print("Z = ", z_src, " OII SNR = ", oii_3726_snr, oii_3729_snr)
        except:
            line_snr = 0
            print("Z = ", z_src, " WARNING: no [OII] SNR")
    else:
        try:
            line_snr = get_line_feature(src, line= line, feature="SNR")
            print("Z = ", z_src," ", line, " SNR = ", line_snr)
        except:
            line_snr = 0
            print("Z = ", z_src, " WARNING: no ", line, " SNR")
       

    if z_src <= zmax_oii and line_snr >= snr_min:

        os.makedirs(src_output_path, exist_ok=True)

        # We open the line table from the source:
        t = src.tables["PL_LINES"]
    
        # then we open the cube and save a copy.
        cube = src.cubes["MUSE_CUBE"]
        cube.write(cube_output_path)


        # We proceed to the continuum substraction:
        # the wavelength of the observed lines:
        if line == "OII":
            tt1 = t[t["LINE"] == "OII3726"]
            tt2 = t[t["LINE"] == "OII3729"]
            L_oii_1_obs = tt1["LBDA_REST"][0] * (1 + tt1["Z"][0])
            L_oii_2_obs = tt2["LBDA_REST"][0] * (1 + tt2["Z"][0])
            L_central_rest = (tt1["LBDA_REST"][0] + tt2["LBDA_REST"][0])/2
            L_central = (L_oii_1_obs + L_oii_2_obs) /2
        else:
            tt = t[t["LINE"] == line]
            f1 = tt["FAMILY"] == "all"
            f2 = tt["FAMILY"] == "balmer"
            tt = tt[f1 | f2]
            L_central_rest = tt["LBDA_REST"][0]
            L_central = tt["LBDA_REST"][0] * (1 + tt["Z"][0])
            
        # the corresponding pixel is:
        central_pix = int(np.round(cube.wave.pixel(L_central)))
        min_pix = central_pix - 16
        max_pix = central_pix + 15
        nb_min_pix = central_pix - 9 # for the narrow band image
        nb_max_pix = central_pix + 8 # for the narrow band image

        # the pixel index of the left continuum
        left_min = central_pix - 150
        right_max =  central_pix + 150


        # the left, right and central cube:
        cube_left = cube[left_min: min_pix, :, :]
        cube_right = cube[max_pix: right_max, :, :]
        cube_central = cube[min_pix: max_pix, :, :]
        cube_central_nb = cube[nb_min_pix: nb_max_pix, :, :]

        # the continuum estimation:
        cont_left = cube_left.mean(axis=0)
        cont_right = cube_right.mean(axis=0)
        cont_mean = 0.5 * (cont_left + cont_right)

        # continuum substraction:
        cube_central_nocont = cube_central - cont_mean
        cube_central_nb_nocont = cube_central_nb - cont_mean
        
        # We get the source mask:
        mask_obj = src.images["MASK_OBJ"]
        ma = mask_obj.data
        mask = np.ma.getdata(ma)
        l = cube.shape[0]
        l_central = cube_central.shape[0]
        l_nb = cube_central_nb.shape[0]
        M = np.repeat(mask[np.newaxis, :, :], l, axis=0) # the 3D mask for the cube
        M_central = np.repeat(mask[np.newaxis, :, :], l_central, axis=0) # the 3D mask for the line cube
        M_nb = np.repeat(mask[np.newaxis, :, :], l_nb, axis=0) # the 3D mask for the narrow band cube
        
        # the SNR estimation:
        try:
            noise_left = (cube_left.data).std(axis = 0)
            noise_right = (cube_right.data).std(axis = 0)
            avg_noise = 0.5*(noise_left + noise_right) # the map of noise per pixel
            signal = (cube_central_nocont.data).max(axis = 0)
            snr = signal/avg_noise # this is a map of SNR per pixel.
            snr_source = snr*mask_obj
            snr_max = snr_source.max() #the maximum snr among pixels
            # we save the SNR values in a text file that we will be able to read later:
        except:
            print("SNR max calc failed !")
            snr_max = 0
            
        fsf = src.get_FSF()
        psf_fwhm = fsf.get_fwhm(L_central)
        psf_beta = fsf.get_beta(L_central)
        lsf_fwhm = (5.835e-8) * L_central ** 2 - 9.080e-4 * L_central + 5.983  # from Bacon 2017
        d = np.array([L_central_rest, L_central, psf_fwhm, psf_beta, lsf_fwhm, line_snr, snr_max])
        col = ["wave_rest", "wave_obs","psf_fwhm", "psf_beta", "lsf_fwhm", "snr_from_src", "snr_max"]
        df = pd.DataFrame(data = [d], columns = col)
        df.to_csv(src_output_path+"/"+line+"_snr.txt", index = False)

        # We make a pdf of the continuum substration:
        pdf_name = src_output_path+"/"+ src_name+"_"+ line +"_continuum_sub.pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)


        ima_central = cube_central.sum(axis=0)
        ima_central_nb_nocont = cube_central_nb_nocont.sum(axis=0)
        cube_central_nb_nocont_masked = cube_central_nb_nocont * M_nb
        ima_central_nb_nocont_masked = cube_central_nb_nocont_masked.sum(axis=0)
        title = src_name + " z = " + str(z_src)
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle(title)

        plt.subplot(221)
        plt.title("before substraction")
        ima_central.plot(scale='arcsinh', colorbar='v', vmin=0, vmax=100)

        plt.subplot(222)
        title_after = "after substraction "+ line+" snr = " + str(np.round(line_snr, 2)) + " SNRmax = "+ str(np.round(snr_max, 2))
        plt.title(title_after)
        ima_central_nb_nocont.plot(scale='arcsinh', colorbar='v', vmin=0, vmax=100)

        plt.subplot(223)
        plt.title("full spectrum")
        plt.axvline(L_central, color="black", linestyle="--", alpha=0.3)
        plt.axvline(L_central - 20 - 150, color="lightgray", linestyle=":")
        plt.axvline(L_central - 20, color="lightgray", linestyle=":")
        plt.axvline(L_central + 20 + 150, color="lightgray", linestyle=":")
        plt.axvline(L_central + 20, color="lightgray", linestyle=":")
        cube_masked = cube*M
        sp = cube_masked.sum(axis = (1,2))
        #sp = cube_masked[:, 15, 15]
        sp.plot()
        plt.axhline(0, color="red")

        plt.subplot(224)
        title = line +" continuum substracted"
        plt.title(title)
        cube_central_nocont_masked = cube_central_nocont*M_central
        sp = cube_central_nocont_masked.sum(axis = (1,2))
        #sp = cube_oii_nocont[:, 15, 15]
        sp.plot()
        plt.axhline(0, color="red")

        cube_nocont_output_path = output_path_field + src_name + "/" + src_name +"_"+ line +"_cube_nocont.fits"
        cube_central_nocont.write(cube_nocont_output_path)
        pdf.savefig(fig)

        pdf.close()

        return fig

    return 0



#-------------------------------------------------------------------------

def substract_all_continuum(field_list, input_path, output_path, snr_min = 15, line = "OII"):
    """
    extract the continuum around the oii line for all the sources in the input folder
    """

    for field in field_list:
        input_path_field = input_path + field + "/" + "products/sources/"
        output_path_field = output_path + field + "/"
        src_file_list = os.listdir(input_path_field)
        os.makedirs(output_path_field, exist_ok=True)

        # we also create a pdf that summarize the continuum substraction:
        pdf_name = field + "_substracted_cube.pdf"
        pdf_path = output_path_field + pdf_name
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

        for src_file in src_file_list:
            print(src_file)
            if "source" in src_file:
                src_input_path = input_path_field + src_file
                src_output_path = output_path_field
                try:    
                    fig = substract_continuum(src_input_path, output_path, snr_min = snr_min, line = line)
                    if fig != 0:
                        pdf.savefig(fig)
                except:
                    print(" !!!!!!!!!!!!!!! CONTINUUM SUBSTRACTION FAILED !!!!!!!!!!!!!!!!!!")

        pdf.close()

    return


#-----------------------------------------------------------------------------

def run_galpak(src_path, output_path, flux_profile = "sersic", rotation_curve = "tanh", thickness_profile = "gaussian", autorun = False, save = False, overwrite = True, line = "OII", fsf = None, **kwargs):
    """
    run galpak for a single source
    line could be OII, OIII, Ha, or cont to run on continuum.
    the fsf could be specified by giving an array like [psf_fwhm, psf_beta, lsf_fwhm]
    """

    # First we open the source:
    src = Source.from_file(src_path)


    src_name = src_path[-28:-5] # the source name in the format "JxxxxXxxxx_source_xxxxx"
    field =  src_path[-28:-18] # the field id in the format JxxxxXxxxx"
    save_name = "run_"+line


    output_path_field = output_path + field + "/"
    src_output_path = output_path_field + src_name+ "/"
    #cube_nocont = Cube(cube_nocont_path)
    
    output_run_list = os.listdir(src_output_path)

    if overwrite == False:
        if save_name in output_run_list:
            print("SKIP job: this run already exists")
            return
    
    # We extract the redshift of the source:
    z_src = src.z["Z"][0]
    print("z = ", z_src)

    # the observed line wavelengths:
    #L_oii_1_obs = L_oii_1 * (1 + z_src)
    #L_oii_2_obs = L_oii_2 * (1 + z_src)
    
        # we configure the instrument with the PSF & LSF from the source file:
    #fsf = src.get_FSF()
    #instru =  galpak.MUSEWFM()
    #instru.psf = MoffatPointSpreadFunction( \
    #    fwhm=fsf.get_fwhm(L_oii_1_obs), \
    #    beta=fsf.get_beta(L_oii_1_obs))
    #instru.lsf = GaussianLineSpreadFunction( \
    #    fwhm=(5.835e-8) * L_oii_1_obs ** 2 - 9.080e-4 * L_oii_1_obs + 5.983)  # from Bacon 2017
    
    FSF_info_path = src_output_path + line +"_snr.txt"
    FSF_info = pd.read_csv(FSF_info_path)
    instru =  galpak.MUSEWFM()
    if fsf == None:
        instru.psf = MoffatPointSpreadFunction( \
            fwhm= FSF_info["psf_fwhm"][0], \
            beta= FSF_info["psf_beta"][0])
        instru.lsf = GaussianLineSpreadFunction(fwhm= FSF_info["lsf_fwhm"][0])
    else:
        instru.psf = MoffatPointSpreadFunction( \
            fwhm= fsf[0], \
            beta= fsf[1])
        instru.lsf = GaussianLineSpreadFunction(fwhm= fsf[2])
    
    # we define the model:
    if line  != "cont":
        model = galpak.DiskModel(flux_profile = flux_profile, rotation_curve= rotation_curve, redshift=z_src, thickness_profile = thickness_profile)
        #pmin = model.Parameters()
        #pmax = model.Parameters()
        #pmin.x = 5
        #pmin.y = 5
        #pmin.z = 5
        #pmin.radius = 2
        #pmax.x = 25
        #pmax.y = 25
        #pmax.z = 25
        #pmax.radius = 20
        
        cube_nocont_path = src_output_path + src_name + "_"+ line+ "_cube_nocont.fits"
        cube_nocont = Cube(cube_nocont_path)
        cube_to_fit = cube_nocont
    else:
        model = galpak.ModelSersic2D(flux_profile = flux_profile, redshift=z_src)
        cube_cont_path_list = glob.glob(src_output_path + src_name + "_cube_cont*.fits")
        cube_cont_path = cube_cont_path_list[0]
        cube_cont = Cube(cube_cont_path)
        cube_to_fit = cube_cont
        
        
    # Then we run galpak:
    if autorun == True:
        gk = galpak.autorun(cube_to_fit, model=model, instrument=instru, **kwargs)
    else:
        gk = galpak.GalPaK3D(cube_to_fit, model=model, instrument=instru)
        
        gk.run_mcmc(**kwargs)
        #gk.run_mcmc(min_boundaries = pmin, max_boundaries = pmax, **kwargs)
        
        

    #gk = galpak.run(cube_nocont, instrument=instru, **kwargs)

    if save == True:
        # We save the galpak files in a dedicated output folder
        galpak_output_path = output_path_field + src_name + "/"+save_name+"/"
        os.makedirs(galpak_output_path, exist_ok=True)
        galpak_output_name = galpak_output_path + "run"
        gk.save(galpak_output_name, overwrite=True)
        print(galpak_output_path)

    return


#------------------------------------------------------------------------------------------

def run_galpak_all(input_path, output_path, field_list, snr_min = 15, mag_sdss_r_max = 26,\
                   flux_profile = "sersic", rotation_curve = "tanh",autorun = False,\
                   save = False, line = "OII", overwrite = True, **kwargs):
    """
    run galpak for all the source
    """
    for field in field_list:
        #pdf_name = field+"_galpak_"+ save_name +".pdf"
        #pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
        
        print(" ")
        print(" ")
        print(" ")
        print(" ")
        print(" FIELD : ", field)
        print(" ")
        print(" ")
        print(" ")
        print(" ")
        
        input_path_field = input_path + field + "/" +"products/sources/"
        output_path_field = output_path + field + "/"
      
        src_file_list = os.listdir(input_path_field)
        
        for src_file in src_file_list:
            if "source" in src_file:
                print(src_file)
                src_path = input_path_field + src_file
                src_name = src_file[:-5]
                src_output_path = output_path_field + src_name+ "/"

                # First we open the source:
                #print(src_path)
                src = Source.from_file(src_path)

                # we get the SNR:
                try:
                    #print(src_output_path+"oii_snr.txt")
                    oii_snr_df = pd.read_csv(src_output_path+"oii_snr.txt", sep = ",", index_col= None)
                    oii_snr_s = oii_snr_df.squeeze()
                    #print(oii_snr_s)
                    oii_3726_snr_from_src = oii_snr_s["snr_3726_from_src"]
                    oii_3729_snr_from_src = oii_snr_s["snr_3729_from_src"]
                    oii_snr_max = oii_snr_s["snr_max"] 
                    #oii_snr = get_line_feature(src, line="OII3726", feature="SNR")
                    print("OII SNR from src = ", oii_3726_snr_from_src, oii_3729_snr_from_src, "  max = ", oii_snr_max)
                except:
                    oii_3726_snr_from_src = 0
                    oii_snr_max = 0
                    print("WARNING: No oii SNR")

                #print("snr_min = ", snr_min)
                if line != "cont":
                    if max(oii_3726_snr_from_src, oii_3729_snr_from_src) >= snr_min:
                        print("**** RUN GALPAK")
                        try: 
                            run_galpak(src_path, output_path, \
                                       flux_profile = flux_profile, rotation_curve = rotation_curve, autorun = autorun,\
                                   save = save, line = line, overwrite = overwrite,\
                                       **kwargs)
                        except:
                            print(" !!!! RUN FAILED !!!!")
                            
                else:
                    try:
                        t = src.tables["SPECPHOT_DR2"]
                        tt = t[t["ID"] == src.header["ID"]]
                        sdss_r = tt["mag_SDSS_r"][0]
                    except:
                        sdss_r = 99
                        print("WARNING: No SDSS mag")
                        
                    if sdss_r <= mag_sdss_r_max:
                        print("**** RUN GALPAK")
                        try: 
                            run_galpak(src_path, output_path, \
                                       flux_profile = flux_profile, rotation_curve = rotation_curve, autorun = autorun,\
                                   save = save, line = line, overwrite = overwrite,\
                                       **kwargs)
                        except:
                            print(" !!!! RUN FAILED !!!!")

    return

# -------------------------------------------------------

def extract_result_single_run(run_name, src_output_path):
    gal_param_ext = "galaxy_parameters.dat"
    convergence_ext = "galaxy_parameters_convergence.dat"
    stats_ext = "stats.dat"
    model_ext = "model.txt"
    line_snr_ext = "*_snr.txt"
    run_split = run_name.split("_") # we split the run name to get the line
    line_name = run_split[1]
    
    run_path = src_output_path + run_name + "/"
    run_file_list = os.listdir(run_path)
    src_file_list = os.listdir(src_output_path)
    #print(run_file_list)
    for f in run_file_list:
        if gal_param_ext in f:
            gal_param_path = run_path + f
        if convergence_ext in f:
            convergence_path = run_path + f
        if stats_ext in f:
            stats_path = run_path + f
        if model_ext in f:
            model_path = run_path + f
    if line_name != "cont":
        try:
            line_snr_path = src_output_path + line_name + "_snr.txt"
            #line_snr_path = glob.glob(src_output_path + line_snr_ext)[0] # WARNING move the oii_snr.txt in the run directory..
        except:
            print("no snr file")
            line_snr_path = ""
    
    gal_param_df = pd.read_csv(gal_param_path, sep = "|", index_col= None)
    gal_param_df.columns = gal_param_df.columns.str.strip()
    convergence_df = pd.read_csv(convergence_path, sep = "|", index_col= None)
    convergence_df.columns = convergence_df.columns.str.strip()
    stats_df = pd.read_csv(stats_path, sep = "|", index_col= None)
    stats_df.columns = stats_df.columns.str.strip()
    model_df = pd.read_csv(model_path, sep = "=", index_col= None)
    model_df = model_df.T
    model_df = model_df.reset_index(drop = True)
    model_df.columns = model_df.columns.str.strip()
    model_df.loc[0].str.strip()
    if line_name != "cont":
        line_snr_df = pd.read_csv(line_snr_path, sep = ",", index_col= None)
    
    #--- for the galaxy parameters file ------
    try:
        gal_param_cols = ["x", "y", "z", "flux", "radius", "sersic_n", "inclination", "pa", "turnover_radius", \
                         "maximum_velocity", "velocity_dispersion"]
        gal_param_error_cols = [col + "_err"  for col in gal_param_cols]
        gal_param = gal_param_df[gal_param_cols]
    except:
        gal_param_cols = ["x", "y", "z", "flux", "radius", "sersic_n", "inclination", "pa"]
        gal_param_error_cols = [col + "_err"  for col in gal_param_cols]
        gal_param = gal_param_df[gal_param_cols]
        
    gal_param_values = np.array(gal_param.loc[0])
    gal_param_errors = np.array(gal_param.loc[1])

    
    gal_param_data = np.array(list(gal_param_values) + list(gal_param_errors))
    gal_param_all_cols = np.array(gal_param_cols + gal_param_error_cols)
    
    gal_param_results = pd.DataFrame(data = [gal_param_data], columns = gal_param_all_cols)
    
    #--- for the galaxy parameters convergence file ------
    convergence_cols = gal_param_cols
    convergence = convergence_df[convergence_cols]
    convergence.columns = convergence.columns + "_convergence"
    
    #--- for the stats file ------
    stats_cols = ["best_chi2","chi2_at_p","BIC","Ndegree","BICr","Ndegree_r","Nr","AIC","k","pD","DIC","SNRmax"]
    stats = stats_df[stats_cols]
    
    #--- for the model file ------
    model = model_df
    
    #--- for the SNR file --------
    if line_name != "cont":
        line_snr = line_snr_df

    #print(oii_snr)    
    
    # build the final dataframe:
    if line_name != "cont":
        DF = pd.concat([gal_param_results, convergence, stats, model, line_snr], axis = 1)
    else:
        DF = pd.concat([gal_param_results, convergence, stats, model], axis = 1)
    
    # compute the corresponding SNR eff:
    #muse_sampling = 0.2 #arcesc per pixel
    #R["snr_eff"] = R["snr_max"]*R["radius"]*muse_sampling/R["psf_fwhm"]
    
    return DF




# -------------------------------------------------------

def extract_results_all(output_path):
    """
    extract all the results for all the field of the output directory and build a table with a line per source and per run.
    """
    results_list = []
    field_list = os.listdir(output_path)
    for f in field_list:
        field_output_path = output_path + f + "/"
        if os.path.isdir(field_output_path):
            src_list = os.listdir(field_output_path)
            print(f)
            for s in src_list:
                src_output_path = field_output_path + s +"/"
                if os.path.isdir(src_output_path):
                    run_list = os.listdir(src_output_path)
                    print(" ",s)
                    for r in run_list:
                        run_output_path = src_output_path + r +"/"
                        if os.path.isdir(run_output_path) and ("run" in r) and (len(os.listdir(run_output_path))!=0):
                            print("    ",r)
                            try:
                                df = extract_result_single_run(r, src_output_path)
                                df.insert(0, "field_id", [f])
                                df.insert(1, "source_id", [s[-5:]])
                                df.insert(2, "run_name", [r])
                                results_list.append(df)
                            except:
                                print("  !!! EXTRACTION FAILED !!!")

    Results = pd.concat(results_list, ignore_index= True)
    
    # Finally we compute the SNR eff for
    muse_sampling = 0.2 #arcesc per pixel
    """ 
    What we want to do is to compute the SNR effective for a source (see Bouche 2015 and Bouche 2021). The SNR eff is useful to select sources because it takes into account the spatial extent of the source and the PSF. Indeed if the source is very small (on few pixels) we need a high OII SNR to deduce the kniematic. At the contrary if we have many pixels, even if the SNR per pixel is low, we can get a quite precise overall idea of the kinematics.
    """
    Results["snr_eff"] = Results["snr_max"]*Results["radius"]*muse_sampling/Results["psf_fwhm"]
    print("Number of run extracted = ", len(Results))
    
    # Finally we rename the ID column to be consistent with other catalogs:
    Results = Results.rename(columns={"source_id": "ID"})
    Results["ID"] = Results["ID"].astype(int)
    
    return Results


#---------------------------------------------------------

def delete_empty_runs(path, field_list):
    for f in field_list:
        field_output_path = path + f + "/"
        if os.path.isdir(field_output_path):
            src_list = os.listdir(field_output_path)
            print(f)
            for s in src_list:
                src_output_path = field_output_path + s +"/"
                if os.path.isdir(src_output_path):
                    run_list = os.listdir(src_output_path)
                    print(" ",s)
                    for r in run_list:
                            run_output_path = src_output_path + r +"/"
                            if os.path.isdir(run_output_path):
                                #print("    ",r)
                                flist = os.listdir(run_output_path)
                                if flist == []:
                                    print("EMPTY DIR")
                                    os.rmdir(run_output_path)
    return

# -----------------------------------------------

def match_results_with_catalogs(galpak_res, dr2_path, fields_info_path, Abs_path, primary_tab_path, output_path = "", \
                               media_path = "", dv_abs_match = 0.5e6, export = False, export_name = "results.csv"):
    
    # First we match the galpak results with the DR2:
    dr2 = format_DR2(dr2_path)
    
    # Then we match with the QSOs from field info:
    dr2 = match_qso(dr2, fields_info_path)
    
    # Then we open the absorptions:
    Abs = pd.read_csv(Abs_path)
    # We compute the Nb of galaxies per Abs:
    
    Abs = get_Nxxx_abs(Abs, dr2, bmax = 2000, dv = dv_abs_match)
    Abs = get_Nxxx_abs(Abs, dr2, bmax = 100, dv = dv_abs_match)
    # and we match the galaxies with the absorbers
    dr2 = match_absorptions_isolated_galaxies(dr2, Abs, dv = dv_abs_match)
    
    # We compute the number of neighbour for each galaxy:
    dr2 = get_Nxxx_neighb(dr2, radius = 150, dv = dv_abs_match) 
    dr2 = get_Nxxx_neighb(dr2, radius = 100, dv = dv_abs_match)
    dr2 = get_Nxxx_neighb(dr2, radius = 50, dv = dv_abs_match)
    
    # We identify the closest galaxy and it's distance:
    dr2 = identify_closest_neighbour(dr2)
    
    # We compute the number of neighbour around the LOS:
    dr2 = get_Nxxx_LOS_all(dr2, bmax = 100, dv = dv_abs_match)
    dr2 = get_Nxxx_LOS_all(dr2, bmax = 2000, dv = dv_abs_match)
    
    # Then we match with the galpak results:
    R = match_DR2(galpak_res, dr2)
    # We compute the alpha parameter:
    R = compute_alpha(R)
    
    # We read the primary & score tab:
    R = read_primary_and_scores(R, primary_tab_path)
    
    # We give the adresses of the runs (to make hyperlink easily):
    R["address_link"] = output_path+R["field_id"]+"/"+R["field_id"]+"_source-"+R["ID"].astype(str)
    R["address_perso"] = media_path+R["field_id"]+"/"+R["field_id"]+"_source-"+R["ID"].astype(str)

    
    # and we export the result:
    if export:
        R.to_csv(export_name, index = False)
    
    return R


# ------------------------------------------

def format_DR2(dr2_path):
    """
    Read the DR2 catalog and format it correctly.
    """
    # first we open the DR2 and re format the columns to be consistent with other scripts:
    temp = Table.read(dr2_path, format='fits')
    data = temp.to_pandas()
    data = data.astype({'FIELD': 'string'})
    data = data.rename(columns={"FIELD": "field_id"})
    data = data.rename(columns={"white_ID": "WHITE_ID"})
    f1 = data["Z"].isnull() == False
    f2 = data["Z"] != 0
    f3 = data["ZCONF"] != 0
    df = data[f1 & f2 & f3]
    
    fields_list = df["field_id"].unique()
    for f in fields_list:
        idx = df.index[df["field_id"] == f].tolist()
        df.loc[idx, "field_id"] = f[2:12]
    return df

#------------------------------------------------------------

def match_DR2(galpak_res, dr2):
    """
    match the catalog of the galpak runs to the DR2 catalog.
    """
    print("Nb of row in the DR2 catalog: ", len(dr2))
    R = pd.merge(dr2, galpak_res, how="left", on=["field_id", "ID"])
    return R

#---------------------------------------------------
def match_qso(df, fields_info_path):
    fields_info = pd.read_csv(fields_info_path)
    fields_info = fields_info.rename({'PSF': 'PSF_qso', 'Comments': 'Comments_qso', 'depth': "depth_qso", \
                                    'ebv_sandf': 'ebv_sandf_qso', 'ebv_planck': 'ebv_planck_qso', \
                                'ebv_sfd': 'ebv_sfd_qso', "HST": "HST_qso", 'rmag': 'rmag_qso' }, axis='columns')
    qso_sub = fields_info[["field_id", 'EXPTIME(s)','PSF_qso',\
                        'Comments_qso', 'zqso_sdss', 'depth_qso',\
                    'ebv_sfd_qso', 'ebv_sandf_qso', 'ebv_planck_qso', 'HST_qso', 'rmag_qso']]


    df = df.merge(qso_sub, on = "field_id", how = "left")
    
    ra = fields_info["ra"]
    dec = fields_info["dec"]

    c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    fields_info["ra_qso"] = c.ra.value
    fields_info["dec_qso"] = c.dec.value
    
    ra = []
    dec = []
    for i, g in df.iterrows():
        f = g["field_id"]
        qso = fields_info[fields_info["field_id"] == f]
        ra.append(qso["ra_qso"])
        dec.append(qso["dec_qso"])

    ra_qso = np.array(ra)
    dec_qso = np.array(dec)
    #print(ra_qso)
    df["ra_qso"] = ra_qso
    df["dec_qso"] = dec_qso
    deg_to_rad = (1*u.degree).to(u.radian).value
    
    return df

#--------------------------------------------------------
def compute_alpha(R):
        
    ad = R["dec_qso"] - R["DEC"]
    op = (R["ra_qso"] - R["RA"])*np.cos(2*np.pi*R["DEC"]/360)

    theta = np.arctan(op/ad)
    theta = theta*360/2/np.pi
 
    alpha = theta%180 - R["pa"]%180
    alpha = (alpha - 90)%180 - 90
    alpha = np.abs(alpha)
    R["alpha"] = alpha
    
    return R

#-----------------------------------------------------------

def match_absorptions_isolated_galaxies(R, Abs, dv = 0.5e6):
    """
    match the Abs dataframe describing the absorptions with the R dataframe containing the galaxies.
    
    dv: maximum velocity difference used to appariate an absorption with a galaxy
    """
   
    pd.options.mode.chained_assignment = None
    
    R["REW_2796"] = 0
    R["sig_REW_2796"] = 0
    R["z_absorption"] = 0
    R["z_absorption_dist"] = 0
    R["N100_abs"] = 0
    R["N2000_abs"] = 0
    for j,i in R.iterrows():
        T = Abs[Abs["field_name"] == i["field_id"]]
        T["v_dist"] = abs(T["z_abs"] - i["Z"])*const.c.value/(1+i["Z"])
        min_v_dist = T["v_dist"].min()
        abs_min = T[T["v_dist"] == min_v_dist]
        idx = R.index[R["ID"]== i["ID"]].to_list()[0]
        R.loc[idx, "REW_2796" ] = abs_min["REW_2796"].mean()
        R.loc[idx, "N100_abs"] = abs_min["N100_abs"].mean()
        R.loc[idx, "N2000_abs"] = abs_min["N2000_abs"].mean()
        try:
            R.loc[idx, "sig_REW_2796"] = abs_min["sig_REW_2796"].mean()
        except:
            print("error")
        R.loc[idx, "z_absorption"] = abs_min["z_abs"].mean()
        R.loc[idx, "vel_absorption_dist"] = min_v_dist
        

    R["bool_absorption"] = 0
    idx = R.index[R["vel_absorption_dist"] < dv].to_list()
    R.loc[idx, "bool_absorption" ] = 1

    R["REW_2796"] = R["bool_absorption"]*R["REW_2796"]
    R["sig_REW_2796"] = R["sig_REW_2796"]*R["bool_absorption"]
    R["z_absorption"] = R["z_absorption"]*R["bool_absorption"]
    R["vel_absorption_dist"] = R["vel_absorption_dist"]*R["bool_absorption"]
    R["N100_abs"] = R["N100_abs"]*R["bool_absorption"]
    R["N2000_abs"] = R["N2000_abs"]*R["bool_absorption"]

    return R

#--------------------------------------
def get_Nxxx_abs(Abs, R, bmax = 100, dv = 1e6):
    """
    Get the number of galaxies in a xxxkpc radius around the QSO LOS for each absorber.
    
    dv: maximum velocity difference between the absorption and the galaxies taken into account.
    """
    Nxxx = []
    for i, absorption in Abs.iterrows():
        f1 = np.abs(R["Z"] - absorption["z_abs"])*const.c.value/(1+absorption["z_abs"])<dv
        f2 = R["field_id"] == absorption["field_name"]
        f3 = R["B_KPC"]< bmax
        F = R[f1 & f2 & f3]
        Nxxx.append(len(F))
    colname = "N"+str(bmax)+"_abs"
    Abs[colname] = np.array(Nxxx)
    return Abs

# ---------------------------------------------------
def get_Nxxx_neighb(R, radius = 100, dv = 1e6):
    """
    Get the number of neighbour galaxies within a given radius for each galaxy.
    
    dv: maximum velocity difference taken for taking into account a galaxy
    """
    label = "N"+str(radius)+"_neighb"
    N_neighb = []
    for i, gal in R.iterrows():
        f1 = np.abs(R["Z"] - gal["Z"])*const.c.value/(1+gal["Z"])<dv
        f2 = R["field_id"] == gal["field_id"]
        #f3 = R["ID"] != gal["ID"]
        #F = R[f1 & f2 & f3]
        F = R[f1 & f2]
        #print(F)
        c1 = SkyCoord(gal["RA"]*u.degree, gal["DEC"]*u.degree)
        c2 = SkyCoord(F["RA"]*u.degree, F["DEC"]*u.degree)
        sep = c1.separation(c2)
        F["dist"] = Distance(unit=u.kpc, z = gal["Z"]).value/((1+gal["Z"])**2)
        F["neighb_dist"] = sep.radian*F["dist"]
        F_filt = F[F["neighb_dist"]<radius]
        N_neighb.append(len(F_filt))
    R[label] = np.array(N_neighb)-1
    return R


#--------------------------------------------------
def identify_closest_neighbour(R):
       
    is_closest = []
    b_kpc_neighb = []
    dv = 0.5e6
    for i, g in R.iterrows():
        field = g["field_id"]
        z = g["Z"]
        bool_abs = g["bool_absorption"]
        z_abs = g["z_absorption"]
        if bool_abs:
            k1 = R["field_id"] == field
            k2 = np.abs(R["Z"] - z_abs)*const.c.value/(1+z_abs)<dv
            others = R[k1 & k2]
            others.sort_values(by = "B_KPC", inplace = True, ignore_index = True)
            #print(len(others))
            if np.min(others["B_KPC"]) == g["B_KPC"]:
                is_closest.append(1)
                if len(others) >=2:
                    #print(others["B_KPC"][1] - others["B_KPC"][0])
                    b_kpc_neighb.append(others["B_KPC"][1] - others["B_KPC"][0])
                else:
                    b_kpc_neighb.append(500)
            else:
                is_closest.append(0)
                b_kpc_neighb.append(0)
        else:
            is_closest.append(0)
            b_kpc_neighb.append(0)

    R["is_closest"] = is_closest
    R["B_KPC_NEIGHB"] = b_kpc_neighb
    return R


#------------------------------------

def get_Nxxx_LOS_all(R, bmax = 100, dv = 0.5e6):
    """
    for each galaxy g of R, it computes the number of galaxies in a redshift slice +-dv
    centered on g, at an impact parameter inferior to bmax to the LOS.
    """
    
    label = "N"+str(bmax)+"_LOS"
    N_LOS = []
    for i, g in R.iterrows():
        
        f1 = np.abs(R["Z"] - g["Z"])*const.c.value/(1+g["Z"])<dv
        f2 = R["field_id"] == g["field_id"]
        f3 = R["B_KPC"] <= bmax
        F = R[f1 & f2 & f3]
        N_LOS.append(len(F))
    R[label] = np.array(N_LOS)
    return R

#-----------------------------------------------------
def build_continuum_cube(src_path, output_path, mag_sdss_r_max = 26, L_central = 6750, L_central_auto = True):
    """
    Take a source file as an input, substract the continuum of the cube aroud a given line
    and save the output
    """

    # First we open the source:
    src = Source.from_file(src_path)

    src_name = src_path[-28:-5] # the source name in the format "JxxxxXxxxx_source_xxxxx"
    field =  src_path[-28:-18] # the field id in the format JxxxxXxxxx"

    output_path_field = output_path + field + "/"
    src_output_path = output_path_field + src_name+ "/"
    #cube_output_path = src_output_path + src_name + "_cube.fits"

    # We extract the redshift of the source:
    try:
        z_src = src.z["Z"][0]
    except:
        z_src = 0
        print("WARNING: No redshift")
        
    try:
        t = src.tables["SPECPHOT_DR2"]
        tt = t[t["ID"] == src.header["ID"]]
        sdss_r = tt["mag_SDSS_r"][0]
    except:
        sdss_r = 99
        print("WARNING: No SDSS mag")
        
        
    if sdss_r <= mag_sdss_r_max:

        os.makedirs(src_output_path, exist_ok=True)

        # then we open the cube and save a copy.
        cube = src.cubes["MUSE_CUBE"]
        #cube.write(cube_output_path)


        # if L_central auto, we find automatically a good range with no skyline and no other line:
        if L_central_auto:
            if z_src < 0.63:
                L_c = 8200
            if z_src >= 0.63 and z_src < 0.9:
                L_c = 5400
            if z_src >= 0.9:
                L_c = 7150
        else: 
            L_c = L_central
        
        #L_central = 6750

        # the corresponding pixel is:
        print("L central = ", L_c)
        central_pix = int(np.round(cube.wave.pixel(L_c)))
        min_pix = central_pix - 16
        max_pix = central_pix + 15

        # the continuum cube:
        cube_cont = cube[min_pix: max_pix, :, :]

        # We get the source mask:
        mask_obj = src.images["MASK_OBJ"]
        ma = mask_obj.data
        mask = np.ma.getdata(ma)
        l = cube_cont.shape[0]
        M = np.repeat(mask[np.newaxis, :, :], l, axis=0) # the 3D mask for the continuum cube
        cube_cont_masked = cube_cont * M
        ima_cont = cube_cont.sum(axis=0)
        spe_cont = cube_cont_masked.sum(axis= (1,2))
        spe_cont_mean = cube_cont_masked.mean(axis= (1,2))

        # the continuum estimation:
        #continuum = cube_cont_masked.mean(axis = (0,1,2))
        continuum = cube_cont_masked.mean()


        #fsf = src.get_FSF()
        #psf_fwhm = fsf.get_fwhm(L_oii_1_obs)
        #lsf_fwhm = (5.835e-8) * L_oii_1_obs ** 2 - 9.080e-4 * L_oii_1_obs + 5.983  # from Bacon 2017
        #d = np.array([psf_fwhm, lsf_fwhm, oii_3726_snr, oii_3729_snr, snr_max])
        #col = ["psf_fwhm", "lsf_fwhm", "snr_3726_from_src","snr_3729_from_src", "snr_max"]
        #df = pd.DataFrame(data = [d], columns = col)
        #df.to_csv(src_output_path+"/"+"oii_snr.txt", index = False)

        # We make a pdf of the continuum:
        pdf_name = src_output_path+"/"+ src_name+"_continuum_" + str(np.round(L_c,0))+".pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)



        title = src_name + " z = " + str(np.round(z_src,4))
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle(title)

        plt.subplot(221)
        title_cont = "continuum at " +  str(np.round(L_central,0))+ " = " + str(np.round(continuum*1000, 2)) +"e-20 erg/A/s/cm2/"
        plt.title(title_cont)
        ima_cont.plot(scale='arcsinh', colorbar='v', vmin=0, vmax=100)

        plt.subplot(222)
        title_sp = "continuum spectrum - SDSS r = " + str(np.round(sdss_r,2))
        plt.title(title_sp)
        spe_cont.plot()

        cube_cont_path = output_path_field + src_name + "/" + src_name + "_cube_cont" + str(np.round(L_c,0))+".fits"
        cube_cont.write(cube_cont_path)
        pdf.savefig(fig)

        pdf.close()

        return fig

    return 0



#-------------------------------------------------------------------------


def build_all_continuum(field_list, input_path, output_path, mag_sdss_r_max = 26, L_central = 6750, L_central_auto = True):
    """
    extract the continuum around the oii line for all the sources in the input folder
    """

    for field in field_list:
        input_path_field = input_path + field + "/" + "products/sources/"
        output_path_field = output_path + field + "/"
        src_file_list = os.listdir(input_path_field)
        os.makedirs(output_path_field, exist_ok=True)

        # we also create a pdf that summarize the continuum substraction:
        pdf_name = field + "_continuum.pdf"
        pdf_path = output_path_field + pdf_name
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)

        for src_file in src_file_list:
            print(src_file)
            if "source" in src_file:
                src_input_path = input_path_field + src_file
                src_output_path = output_path_field
                #try:    
                fig = build_continuum_cube(src_input_path, output_path, mag_sdss_r_max = mag_sdss_r_max, \
                                           L_central = L_central, L_central_auto = L_central_auto)
                if fig != 0:
                    pdf.savefig(fig)
                #except:
                    #print(" !!!!!!!!!!!!!!! CONTINUUM FAILED !!!!!!!!!!!!!!!!!!")

        pdf.close()

    return

#-----------------------------------------

def build_velocity_map(src_path, output_path, line = "OII", snr_min = 3, commonw=True, dv=500., dxy=15, deltav=2000., initw=50., wmin=30., wmax=250., dfit=100., degcont=0, sclip=10, xyclip=3, nclip=3, wsmooth=0, ssmooth=2.):
    # we open the source file:
    src = Source.from_file(src_path)
    src_name = src_path[-28:-5] # the source name in the format "JxxxxXxxxx_source_xxxxx"
    src_id = int(src_name[-5:])
    field =  src_path[-28:-18] # the field id in the format JxxxxXxxxx"
    output_path_field = output_path + field + "/"
    src_output_path = output_path_field + src_name+ "/"
    camel_path = src_output_path +"camel_"+line
    os.makedirs(camel_path, exist_ok=True)
    
    lines_dict = {"HALPHA":"ha", "OII":"o2", "OIII5007": "o3"}
    
    # We extract the redshift of the source:
    try:
        z_src = src.z["Z"][0]
    except:
        z_src = 0
        print("WARNING: No redshift")
    
    # First we build the catalog needed to run CAMEL:
    dd = [[src.ID, src.ra, src.dec, z_src]]
    df = pd.DataFrame(data = dd, columns = ['ID', 'ra', 'dec', 'z'])
    df.to_csv(camel_path +"/catfile.csv", index=False)

    cube = src.cubes["MUSE_CUBE"]
    #cube.write(cube_output_path)
    
    # we get the SNR:
    # we get the SNR:
    if line == "OII":
        try:
            oii_3726_snr = get_line_feature(src, line="OII3726", feature="SNR")
            oii_3729_snr = get_line_feature(src, line="OII3729", feature="SNR")
            line_snr = max(oii_3726_snr, oii_3729_snr)
            print("Z = ", z_src, " OII SNR = ", oii_3726_snr, oii_3729_snr)
        except:
            line_snr = 0
            print("Z = ", z_src, " WARNING: no [OII] SNR")
    else:
        try:
            line_snr = get_line_feature(src, line= line, feature="SNR")
            print("Z = ", z_src," ", line, " SNR = ", line_snr)
        except:
            line_snr = 0
            print("Z = ", z_src, " WARNING: no ", line, " SNR")
       

    if z_src <= zmax_oii and line_snr >= snr_min:
        
        # We open the line table from the source:
        t = src.tables["PL_LINES"]
        
        # We proceed to the continuum substraction:
        # the wavelength of the observed lines:
        if line == "OII":
            tt1 = t[t["LINE"] == "OII3726"]
            tt2 = t[t["LINE"] == "OII3729"]
            L_oii_1_obs = tt1["LBDA_REST"][0] * (1 + tt1["Z"][0])
            L_oii_2_obs = tt2["LBDA_REST"][0] * (1 + tt2["Z"][0])
            L_central = (L_oii_1_obs + L_oii_2_obs) /2
        else:
            tt = t[t["LINE"] == line]
            L_central = tt["LBDA_REST"][0] * (1 + tt["Z"][0])
            
        # the corresponding pixel is:
        central_pix = int(np.round(cube.wave.pixel(L_central)))
        min_pix = central_pix - 16
        max_pix = central_pix + 15

        # the central cube:
        cube_central = cube[min_pix: max_pix, :, :]
        cube_central_path = src_output_path + src_name + "_"+lines_dict[line]+"_cube.fits"
        cube_central.write(cube_central_path)

        # then we build the needed cubes:
        hdul = fits.open(cube_central_path)
        cube_data = hdul[1].data
        cube_header = hdul[1].header
        hdu = fits.PrimaryHDU(data=cube_data, header = cube_header)
        hdu.writeto(camel_path +"/cube_data.fits", overwrite = True)

        var_data = hdul[2].data
        var_header = hdul[2].header
        hdu = fits.PrimaryHDU(data=var_data, header = var_header)
        hdu.writeto(camel_path +"/var_data.fits", overwrite = True)

        # the parameters to create the config file is then:
        path = camel_path+"/"
        cubefile = cube_central_path
        #cubefile = src_output_path + src_name + "_cube_nocont.fits"
        varfile = camel_path + "/var_data.fits"
        catfile =  "catfile.csv"
        lines = lines_dict[line]
        suffixeout = camel_path+"/camel"

        # we create the configuration file:
        out = cc.create_config(path, cubefile, varfile, catfile, lines, suffixeout, commonw=commonw, dv=dv, dxy=dxy, deltav=deltav, initw=initw, wmin=wmin, wmax=wmax, dfit=dfit, degcont=degcont, sclip=sclip, xyclip=xyclip, nclip=nclip, wsmooth=wsmooth, ssmooth=ssmooth)
        filename = camel_path +"/camel_"+str(src_id)+"_"+lines_dict[line] +".config"
        #/muse/MG2QSO/private/analysis/galpak_dr2/J0014m0028/J0014m0028_source-11122/camel/camel_11122_o2.config
        # then we run camel:
        cml.camel(str(filename), plot=True)

        # Then we create an image with the velocity map:
        # for that we use the mask of the source:

        vel = "/camel_"+ str(src_id) +"_"+lines_dict[line] +"_vel_common.fits"
        snr = "/camel_"+ str(src_id) +"_"+lines_dict[line] +"_snr_common.fits"
        disp = "/camel_"+ str(src_id) +"_"+lines_dict[line] +"_disp_common.fits"
        hdul_vel = fits.open(camel_path+vel)
        hdul_snr = fits.open(camel_path+snr)
        hdul_disp = fits.open(camel_path+disp)
        img_vel = hdul_vel[0].data
        img_snr = hdul_snr[0].data
        img_disp = hdul_disp[0].data
        mask_obj = src.images["MASK_OBJ"]
        ma = mask_obj.data
        m = np.where(img_snr>4, 1, 0)

        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_src).value/60
        extent_arcsec = np.array([-0.2*15, 0.2*15,-0.2*15, 0.2*15])
        extent_kpc = extent_arcsec*kpc_per_arcsec

        pdf_name = src_output_path+"/"+ src_name+"_"+line +"_velmaps.pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)

        fig = plt.figure(figsize = (16,12))
        plt.subplot(2,2,1)
        plt.title("velocity map (src mask)")
        plt.imshow(img_vel*ma, vmin = -150, vmax = 150, cmap = "bwr", extent = extent_kpc)
        plt.xlabel("x [kpc]", size = 12)
        plt.ylabel("y [kpc]", size = 12)
        plt.colorbar()

        plt.subplot(2,2,2)
        plt.title("velocity map (snr>4 mask)")
        plt.imshow(img_vel*m, vmin = -150, vmax = 150, cmap = "bwr", extent = extent_kpc)
        cbar = plt.colorbar(label = "Dv [km/s]")
        plt.xlabel("x [kpc]", size = 12)
        plt.ylabel("y [kpc]", size = 12)

        plt.subplot(2,2,3)
        plt.title("velocity dispersion map (snr>4 mask)")
        plt.imshow(img_disp*m, extent = extent_kpc)
        plt.colorbar(label = "fwhm [km/s]")
        plt.xlabel("x [kpc]", size = 12)
        plt.ylabel("y [kpc]", size = 12)

        pdf.savefig(fig)
        pdf.close()
        return fig
    return 

#----------------------------------------------------------------
def build_velocity_map_all(field_list, input_path, output_path, snr_min = 15, commonw=True, dv=500., dxy=15, deltav=2000., initw=50., wmin=30., wmax=250., dfit=100., degcont=0, sclip=10, xyclip=3, nclip=3, wsmooth=0, ssmooth=0):
    """
    extract the continuum around the oii line for all the sources in the input folder
    """

    for field in field_list:
        input_path_field = input_path + field + "/" + "products/sources/"
        output_path_field = output_path + field + "/"
        src_file_list = os.listdir(input_path_field)
        os.makedirs(output_path_field, exist_ok=True)

        for src_file in src_file_list:
            print(src_file)
            if "source" in src_file:
                src_input_path = input_path_field + src_file
                src_output_path = output_path_field
                try:    
                    fig = build_velocity_map(src_input_path, output_path, snr_min = snr_min, commonw=commonw, dv=dv, dxy=dxy, deltav=deltav, initw=initw, wmin=wmin, wmax=wmax, dfit=dfit, degcont=degcont, sclip=sclip, xyclip=xyclip, nclip=nclip, wsmooth=wsmooth, ssmooth=ssmooth)
                #if fig != 0:
                #    pdf.savefig(fig)
                except:
                    print("VELOCITY MAP FAILED")

        #pdf.close()
    
    return 

#-----------------------------------------------------------------

def read_primary_and_scores(R, primary_path):
    """
    read a recap tab containing the following columns:
    ID: the source id
    primary: a 0/1 flag indicating if the galaxy is primary
    
    """
    
    primary = pd.read_csv(primary_path)
    
    R["primary"] = 0
    R["galpak_score"] = 0
    
    for i, r in primary.iterrows():
        idx = R.index[R["ID"]== r["ID"]].to_list()
        R.loc[idx, "primary"] =  r["primary"]
        R.loc[idx, "galpak_score"] =  r["score"]

    
    return R

#------------------------------------------
def read_runs(R, runs_path):
    """
    small script to read from a csv file which run is the one to use for each galaxy.
    """
    runs = pd.read_csv(runs_path)
    
    for i, r in runs.iterrows():
        f1 = R["ID"]== r["ID"]
        f2 = R["run_name"]== r["run_name"]
        idx = R.index[f1 & f2].to_list()
        R.loc[idx, "current"] =  r["to keep"]

    return R









