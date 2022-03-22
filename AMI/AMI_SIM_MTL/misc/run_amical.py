from pathlib import Path

import amical
from astropy.io import fits
from matplotlib import pyplot as plt

# =============================================================================
# Constants
# =============================================================================
# Where data is and where oifits will be saved
OUT_DIR = Path("outputs/amwrap/ami_sims/")

# Targets, used to get file
TARGET_NAME = "HD-218396"
CALIB_NAME = "HD-218172"

target_id = TARGET_NAME.replace("-", "_")
calib_id = CALIB_NAME.replace("-", "_")

# Filter, used for dir and file name
FILTER = "F480M"


# Full path to oiits file
oifits_base = f"t_{target_id}_c_{calib_id}.oifits"  # basename for AMICAL
OIFITS_FILE = OUT_DIR / FILTER / oifits_base

# Target and calib file names and full paths
TARGET_FILE = f"t_SKY_SCENE_SIMULATION_1_{target_id}_{FILTER}__psf_{FILTER}_SIMULATION_1_{target_id}_00.fits"
CALIB_FILE = f"t_SKY_SCENE_SIMULATION_1_{calib_id}_{FILTER}__psf_{FILTER}_SIMULATION_1_{calib_id}_00.fits"

file_t = OUT_DIR / FILTER / TARGET_FILE
file_c = OUT_DIR / FILTER / CALIB_FILE

# Show AMICAL plot
SHOW = True

# Whether we should run each step
EXTRACT = True  # Observables extraction
CAL = True  # Calibration
USE_CANDID = True  # chi2 grid
USE_PYMASK = False  # MCMC (being rewritten as of March 2022)

# =============================================================================
# Open data
# =============================================================================
hdu = fits.open(file_t)
cube_t = hdu[0].data
hdu.close()

hdu = fits.open(file_c)
cube_c = hdu[0].data
hdu.close()

# =============================================================================
# Extraction
# =============================================================================
if EXTRACT:
    params_ami = {
        "peakmethod": "fft",
        "bs_multi_tri": False,
        "maskname": "g7",
        "fw_splodge": 0.7,
    }

    bs_t = amical.extract_bs(
        cube_t, file_t, targetname=TARGET_NAME, **params_ami, display=True
    )
    bs_c = amical.extract_bs(
        cube_c, file_c, targetname=CALIB_NAME, **params_ami, display=False
    )

# =============================================================================
# Calibration
# =============================================================================
if CAL:
    cal = amical.calibrate(bs_t, bs_c)

    # Display and save the results as oifits
    if SHOW:
        amical.show(cal)
        dic = amical.save(
            cal,
            datadir=OUT_DIR / FILTER,
            oifits_file=oifits_base,
            fake_obj=False,
        )
        plt.show(block=True)
    else:
        plt.close("all")


# =============================================================================
# Binary fitting
# =============================================================================
if USE_CANDID:
    param_candid = {
        "rmin": 20,  # inner radius of the grid
        "rmax": 500,  # outer radius of the grid
        "step": 50,  # grid sampling
        "ncore": 1,  # core for multiprocessing
    }

    fit1 = amical.candid_grid(
        OIFITS_FILE, **param_candid, diam=20, doNotFit=[], save=False
    )

    # Plot and save the fitted model
    amical.plot_model(OIFITS_FILE, fit1["best"], save=False)

    cr_candid = amical.candid_cr_limit(
        OIFITS_FILE, **param_candid, fitComp=fit1["comp"], save=False
    )

if USE_PYMASK:
    param_pymask = {
        "sep_prior": [20, 500],  # Prior on the separation
        "pa_prior": [-180, 180],  # Prior on the position angle
        "cr_prior": [10, 12000],  # Prior on the contrast ratio
        "ncore": 1,  # core for multiprocessing
        "extra_error_cp": 0,
        "err_scale": 1,
    }

    fit2 = amical.pymask_grid(OIFITS_FILE, **param_pymask)

    param_mcmc = {
        "niters": 800,
        "walkers": 100,
        "initial_guess": [146, 47, 244],
        "burn_in": 100,
    }

    fit3 = amical.pymask_mcmc(OIFITS_FILE, **param_pymask, **param_mcmc)

    cr_pymask = amical.pymask_cr_limit(
        OIFITS_FILE,
        nsim=500,
        ncore=1,
        smax=250,
        nsep=100,
        cmax=5000,
        nth=30,
        ncrat=60,
    )

if USE_CANDID and USE_PYMASK:
    plt.figure()
    plt.plot(cr_candid["r"], cr_candid["cr_limit"], label="CANDID", alpha=0.5, lw=3)
    plt.plot(cr_pymask["r"], cr_pymask["cr_limit"], label="Pymask", alpha=0.5, lw=3)
    plt.ylim(plt.ylim()[1], plt.ylim()[0])  # -- reverse plot
    plt.xlabel("Separation [mas]")
    plt.ylabel(r"$\Delta \mathrm{Mag}_{3\sigma}$")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()

if SHOW:
    plt.show(block=True)
else:
    plt.close("all")
