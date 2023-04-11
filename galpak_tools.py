import numpy as np
import matplotlib.pyplot as plt
import galpak
#from galpak import DefaultModel, ModelSersic

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

def substract_continuum(src_path, output_path, snr_min = 15):
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
    cube_output_path = src_output_path + src_name + "_cube.fits"

    # We extract the redshift of the source:
    try:
        z_src = src.z["Z"][0]
    except:
        z_src = 0
        print("WARNING: No redshift")

    # we get the SNR:
    try:
        oii_3726_snr = get_line_feature(src, line="OII3726", feature="SNR")
        oii_3729_snr = get_line_feature(src, line="OII3729", feature="SNR")
        print("Z = ", z_src, " OII SNR = ", oii_3726_snr, oii_3729_snr)
    except:
        oii_snr = 0
        print("Z = ", z_src, " WARNING: no [OII] SNR")

    if z_src >= zmin_oii and z_src <= zmax_oii and max(oii_3726_snr, oii_3729_snr) >= snr_min:

        os.makedirs(src_output_path, exist_ok=True)

        # then we open the cube and save a copy.
        cube = src.cubes["MUSE_CUBE"]
        cube.write(cube_output_path)


        # We proceed to the continuum substraction:
        # the wavelength of the observed lines:
        L_oii_1_obs = L_oii_1 * (1 + z_src)
        L_oii_2_obs = L_oii_2 * (1 + z_src)
        L_central_oii = (L_oii_1_obs + L_oii_2_obs) /2

        # the corresponding pixel is:
        central_pix = int(np.round(cube.wave.pixel(L_central_oii)))
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
        cube_oii = cube[min_pix: max_pix, :, :]
        cube_oii_nb = cube[nb_min_pix: nb_max_pix, :, :]

        # the continuum estimation:
        cont_left = cube_left.mean(axis=0)
        cont_right = cube_right.mean(axis=0)
        cont_mean = 0.5 * (cont_left + cont_right)

        # continuum substraction:
        cube_oii_nocont = cube_oii - cont_mean
        cube_oii_nb_nocont = cube_oii_nb - cont_mean
        
        # We get the source mask:
        mask_obj = src.images["MASK_OBJ"]
        ma = mask_obj.data
        mask = np.ma.getdata(ma)
        l = cube.shape[0]
        l_oii = cube_oii.shape[0]
        l_nb = cube_oii_nb.shape[0]
        M = np.repeat(mask[np.newaxis, :, :], l, axis=0) # the 3D mask for the cube
        M_oii = np.repeat(mask[np.newaxis, :, :], l_oii, axis=0) # the 3D mask for the oii cube
        M_nb = np.repeat(mask[np.newaxis, :, :], l_nb, axis=0) # the 3D mask for the narrow band cube
        
        # the SNR estimation:
        try:
            noise_left = (cube_left.data).std(axis = 0)
            noise_right = (cube_right.data).std(axis = 0)
            avg_noise = 0.5*(noise_left + noise_right) # the map of noise per pixel
            signal = (cube_oii_nocont.data).max(axis = 0)
            snr = signal/avg_noise # this is a map of SNR per pixel.
            snr_source = snr*mask_obj
            snr_max = snr_source.max() #the maximum snr among pixels
            # we save the SNR values in a text file that we will be able to read later:
        except:
            print("SNR max calc failed !")
            snr_max = 0
            
        fsf = src.get_FSF()
        psf_fwhm = fsf.get_fwhm(L_oii_1_obs)
        lsf_fwhm = (5.835e-8) * L_oii_1_obs ** 2 - 9.080e-4 * L_oii_1_obs + 5.983  # from Bacon 2017
        d = np.array([psf_fwhm, lsf_fwhm, oii_3726_snr, oii_3729_snr, snr_max])
        col = ["psf_fwhm", "lsf_fwhm", "snr_3726_from_src","snr_3729_from_src", "snr_max"]
        df = pd.DataFrame(data = [d], columns = col)
        df.to_csv(src_output_path+"/"+"oii_snr.txt", index = False)

        # We make a pdf of the continuum substration:
        pdf_name = src_output_path+"/"+ src_name+"_continuum_sub.pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)


        ima_oii = cube_oii.sum(axis=0)
        ima_oii_nb_nocont = cube_oii_nb_nocont.sum(axis=0)
        cube_oii_nb_nocont_masked = cube_oii_nb_nocont * M_nb
        ima_oii_nb_nocont_masked = cube_oii_nb_nocont_masked.sum(axis=0)
        title = src_name + " z = " + str(z_src)
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle(title)

        plt.subplot(221)
        plt.title("before substraction")
        ima_oii.plot(scale='arcsinh', colorbar='v', vmin=0, vmax=100)

        plt.subplot(222)
        title_after = "after substraction,  OII snr = " + str(np.round(oii_3726_snr, 2)) + " SNRmax = "+ str(np.round(snr_max, 2))
        plt.title(title_after)
        ima_oii_nb_nocont.plot(scale='arcsinh', colorbar='v', vmin=0, vmax=100)

        plt.subplot(223)
        plt.title("full spectrum")
        plt.axvline(L_oii_1_obs, color="black", linestyle="--", alpha=0.3)
        plt.axvline(L_oii_1_obs - 20 - 150, color="lightgray", linestyle=":")
        plt.axvline(L_oii_1_obs - 20, color="lightgray", linestyle=":")
        plt.axvline(L_oii_1_obs + 20 + 150, color="lightgray", linestyle=":")
        plt.axvline(L_oii_1_obs + 20, color="lightgray", linestyle=":")
        cube_masked = cube*M
        sp = cube_masked.sum(axis = (1,2))
        #sp = cube_masked[:, 15, 15]
        sp.plot()
        plt.axhline(0, color="red")

        plt.subplot(224)
        plt.title("[OII] lines, continuum substracted")
        cube_oii_nocont_masked = cube_oii_nocont*M_oii
        sp = cube_oii_nocont_masked.sum(axis = (1,2))
        #sp = cube_oii_nocont[:, 15, 15]
        sp.plot()
        plt.axhline(0, color="red")

        cube_nocont_output_path = output_path_field + src_name + "/" + src_name + "_cube_nocont.fits"
        cube_oii_nocont.write(cube_nocont_output_path)
        pdf.savefig(fig)

        pdf.close()

        return fig

    return 0



#-------------------------------------------------------------------------

def substract_all_continuum(field_list, input_path, output_path, snr_min = 15):
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
                    fig = substract_continuum(src_input_path, output_path, snr_min = snr_min)
                    if fig != 0:
                        pdf.savefig(fig)
                except:
                    print(" !!!!!!!!!!!!!!! CONTINUUM SUBSTRACTION FAILED !!!!!!!!!!!!!!!!!!")

        pdf.close()

    return


#-----------------------------------------------------------------------------

def run_galpak(src_path, output_path, flux_profile = "sersic", rotation_curve = "tanh",autorun = False, save = False, save_name = "run1", overwrite = True, continuum_fit = False, **kwargs):
    """
    run galpak for a single source
    """

    # First we open the source:
    src = Source.from_file(src_path)


    src_name = src_path[-28:-5] # the source name in the format "JxxxxXxxxx_source_xxxxx"
    field =  src_path[-28:-18] # the field id in the format JxxxxXxxxx"


    output_path_field = output_path + field + "/"
    src_output_path = output_path_field + src_name+ "/"
    cube_nocont_path = src_output_path + src_name + "_cube_nocont.fits"
    #cube_nocont = Cube(cube_nocont_path)
    
    cube_cont_path_list = glob.glob(src_output_path + src_name + "_cube_cont*.fits")
    cube_cont_path = cube_cont_path_list[0]
    #cube_cont = Cube(cube_cont_path)
    
    output_run_list = os.listdir(src_output_path)

    if overwrite == False:
        if save_name in output_run_list:
            print("SKIP job: this run already exists")
            return
    
    # We extract the redshift of the source:
    z_src = src.z["Z"][0]
    print("z = ", z_src)

    # the observed OII wavelengths:
    L_oii_1_obs = L_oii_1 * (1 + z_src)
    L_oii_2_obs = L_oii_2 * (1 + z_src)

    # we configure the instrument with the PSF & LSF from the source file:
    fsf = src.get_FSF()
    instru =  galpak.MUSEWFM()
    instru.psf = MoffatPointSpreadFunction( \
        fwhm=fsf.get_fwhm(L_oii_1_obs), \
        beta=fsf.get_beta(L_oii_1_obs))
    instru.lsf = GaussianLineSpreadFunction( \
        fwhm=(5.835e-8) * L_oii_1_obs ** 2 - 9.080e-4 * L_oii_1_obs + 5.983)  # from Bacon 2017


    # we define the model:
    if continuum_fit == False:
        model = galpak.DiskModel(flux_profile = flux_profile, rotation_curve= rotation_curve, redshift=z_src, line = galpak.OII)
        cube_nocont = Cube(cube_nocont_path)
        cube_to_fit = cube_nocont
    else:
        model = galpak.ModelSersic2D(flux_profile = flux_profile, redshift=z_src)
        cube_cont = Cube(cube_cont_path)
        cube_to_fit = cube_cont
        
        
    # Then we run galpak:
    if autorun == True:
        gk = galpak.autorun(cube_to_fit, model=model, instrument=instru, **kwargs)
    else:
        gk = galpak.GalPaK3D(cube_to_fit, model=model, instrument=instru)
        gk.run_mcmc(**kwargs)
        
        

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
                   flux_profile = "sersic", rotation_curve = "tanh",autorun = False, continuum_fit = False,\
                   save = False, save_name = "run1", overwrite = True, **kwargs):
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
                if continuum_fit == False:
                    if max(oii_3726_snr_from_src, oii_3729_snr_from_src) >= snr_min:
                        print("**** RUN GALPAK")
                        try: 
                            run_galpak(src_path, output_path, \
                                       flux_profile = flux_profile, rotation_curve = rotation_curve, autorun = autorun,\
                                   save = save, save_name = save_name, continuum_fit = continuum_fit, overwrite = overwrite,\
                                       **kwargs)
                        except:
                            print(" !!!! RUN FAILED !!!!")
                            
                elif continuum_fit == True:
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
                                   save = save, save_name = save_name, continuum_fit = continuum_fit, overwrite = overwrite,\
                                       **kwargs)
                        except:
                            print(" !!!! RUN FAILED !!!!")

    return

# -------------------------------------------------------

def extract_result_single_run(run_path, src_output_path):
    gal_param_ext = "galaxy_parameters.dat"
    convergence_ext = "galaxy_parameters_convergence.dat"
    stats_ext = "stats.dat"
    model_ext = "model.txt"
    oii_snr_ext = "oii_snr.txt"
    
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
    for f in src_file_list: 
        if oii_snr_ext in f:
            oii_snr_path = src_output_path + f
            
    #print(gal_param_path)
    #print(convergence_path)
    #print(stats_path)
    
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
    oii_snr_df = pd.read_csv(oii_snr_path, sep = ",", index_col= None)
    
    #--- for the galaxy parameters file ------
    gal_param_cols = ["x", "y", "z", "flux", "radius", "sersic_n", "inclination", "pa", "turnover_radius", \
                     "maximum_velocity", "velocity_dispersion"]
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
    oii_snr = oii_snr_df

    #print(oii_snr)    
    
    # build the final dataframe:
    DF = pd.concat([gal_param_results, convergence, stats, model, oii_snr], axis = 1)
    
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
                        if os.path.isdir(run_output_path):
                            print("    ",r)
                            try:
                                df = extract_result_single_run(run_output_path, src_output_path)
                                df.insert(0, "field_id", [f])
                                df.insert(1, "source_id", [s[-5:]])
                                df.insert(2, "run_name", [r])
                                results_list.append(df)
                            except:
                                print("  !!! EXTRACTION FAILED !!!")

    Results = pd.concat(results_list, ignore_index= True)
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

def build_velocity_map(src_path, output_path, snr_min = 3, commonw=True, dv=500., dxy=15, deltav=2000., initw=50., wmin=30., wmax=250., dfit=100., degcont=0, sclip=10, xyclip=3, nclip=3, wsmooth=0, ssmooth=2.):
    # we open the source file:
    src = Source.from_file(src_path)
    src_name = src_path[-28:-5] # the source name in the format "JxxxxXxxxx_source_xxxxx"
    src_id = int(src_name[-5:])
    field =  src_path[-28:-18] # the field id in the format JxxxxXxxxx"
    output_path_field = output_path + field + "/"
    src_output_path = output_path_field + src_name+ "/"
    camel_path = src_output_path +"camel"
    os.makedirs(camel_path, exist_ok=True)
    
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
    try:
        oii_3726_snr = get_line_feature(src, line="OII3726", feature="SNR")
        oii_3729_snr = get_line_feature(src, line="OII3729", feature="SNR")
        print("Z = ", z_src, " OII SNR = ", oii_3726_snr, oii_3729_snr)
    except:
        oii_snr = 0
        print("Z = ", z_src, " WARNING: no [OII] SNR")

    if z_src >= zmin_oii and z_src <= zmax_oii and max(oii_3726_snr, oii_3729_snr) >= snr_min:
        # We proceed to the continuum substraction:
        # the wavelength of the observed lines:
        L_oii_1_obs = L_oii_1 * (1 + z_src)
        L_oii_2_obs = L_oii_2 * (1 + z_src)
        L_central_oii = (L_oii_1_obs + L_oii_2_obs) /2

        # the corresponding pixel is:
        central_pix = int(np.round(cube.wave.pixel(L_central_oii)))
        min_pix = central_pix - 16
        max_pix = central_pix + 15

        # the left, right and central cube:
        cube_oii = cube[min_pix: max_pix, :, :]
        cube_oii_path = src_output_path + src_name + "_oii_cube.fits"
        cube_oii.write(cube_oii_path)

        # then we build the needed cubes:
        hdul = fits.open(cube_oii_path)
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
        cubefile = cube_oii_path
        #cubefile = src_output_path + src_name + "_cube_nocont.fits"
        varfile = camel_path + "/var_data.fits"
        catfile =  "catfile.csv"
        lines = "o2"
        suffixeout = camel_path+"/camel"

        # we create the configuration file:
        out = cc.create_config(path, cubefile, varfile, catfile, lines, suffixeout, commonw=commonw, dv=dv, dxy=dxy, deltav=deltav, initw=initw, wmin=wmin, wmax=wmax, dfit=dfit, degcont=degcont, sclip=sclip, xyclip=xyclip, nclip=nclip, wsmooth=wsmooth, ssmooth=ssmooth)
        filename = camel_path +"/camel_"+str(src_id)+"_o2.config"
        #/muse/MG2QSO/private/analysis/galpak_dr2/J0014m0028/J0014m0028_source-11122/camel/camel_11122_o2.config
        # then we run camel:
        cml.camel(str(filename), plot=True)

        # Then we create an image with the velocity map:
        # for that we use the mask of the source:

        vel = "/camel_"+ str(src_id) +"_o2_vel_common.fits"
        snr = "/camel_"+ str(src_id) +"_o2_snr_common.fits"
        disp = "/camel_"+ str(src_id) +"_o2_disp_common.fits"
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

        pdf_name = src_output_path+"/"+ src_name+"_oii_velmaps.pdf"
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

