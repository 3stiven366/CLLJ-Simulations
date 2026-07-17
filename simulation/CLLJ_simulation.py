"""
FluidSim_Barotropic.py
==============================
Two-dimensional barotropic vorticity simulation of Easterly Wave–Caribbean
Low-Level Jet (EW–CLLJ) interactions over the Intra-Americas Seas (IAS).
 
Physical motivation
-------------------
Rivera (2026) documents that the CLLJ provides a barotropically unstable
mean-flow environment for tropical easterly waves during boreal summer. The
Rayleigh–Kuo criterion (∂q̄/∂y changes sign) is satisfied over most of the
OTREC 2019 period, and positive eddy momentum covariance ⟨u′v′⟩ at
Guanacaste and San Andrés indicates mean-to-eddy energy transfer (CBT > 0).
 
This model captures those mechanisms using the 2D barotropic vorticity
equation on a β-plane:
 
    ∂ζ/∂t + J(ψ, ζ + f) = ν₄∇⁴ζ + F
 
where ζ = ∇²ψ is the relative vorticity, f = f₀ + βy is the Coriolis
parameter, ν₄ is the hyperviscosity, and F is the vorticity forcing that
continuously injects easterly-wave energy into the domain.
 
Governing parameters (calibrated to article)
--------------------------------------------
Domain       : 360° × 60° (global zonal, 30°S–30°N)
Resolution   : nx=720, ny=240  →  Δx≈0.5°, Δy≈0.25° (ERA5-equivalent)
β            : 2.29×10⁻¹¹ s⁻¹m⁻¹  (tropical β-plane at ~15°N)
ν₄           : 1×10¹¹ m⁴/s         (scaled from coarser test run)
Δt_max       : 90 s                 (CFL-stable; U_max≈8 m/s, Δx≈55 km)
Integration  : 180 days             (JAS+SON boreal season)
Forcing band : k = 2–7             (covers both EW modes)
 
MPI notes
---------
This script runs identically with and without MPI:
 
    python FluidSim_Barotropic_Cluster.py          # serial
    mpirun -np 16 python FluidSim_Barotropic_Cluster.py  # parallel
 
FluidSim decomposes the domain into horizontal slabs of ny/nproc rows per
rank. ny=240 must be exactly divisible by nproc. Valid choices:
    2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24  (NOT 32: 240/32=7.5)
 
Variable scope under MPI
------------------------
Global (identical across all ranks):
    params, period, m, all physical constants, all functions.
Local (each rank holds its own subdomain slice):
    x_loc, y_loc, u_mean, U, V, rot, omega_fft.
Rank-0 only (NameError on other ranks if accessed):
    oper_coarse, x_c, y_c — used in the forcing function.
 
Reference
---------
Rivera, E.R. (2026). On the Interaction of Tropical Easterly Waves and the
Caribbean Low-Level Jet Using Observed, ERA5 and WWLLN Data over the
Intra-Americas Seas During OTREC 2019. Meteorology, 5(1), 6.
https://doi.org/10.3390/meteorology5010006
"""


from fluidsim.solvers.ns2d.solver import Simul
from fluiddyn.util.mpi import rank
import numpy as np
import time 

t_inicio = time.time()
# ============================================================================
#  SECTION 1 — GLOBAL CONSTANTS 
# ============================================================================

np.random.seed(42)
m: float = 111e3                        # Degrees to meters
days: int = 90                          # Days of simulation
period: int = 86400 * days              # [s]

ACTIVE_FORCING = True                   # True / False


Lx_deg: int = 120                       # [°]  zonal domain (100°W - 60°W)
Ly_deg: int = 20                        # [°] southern domain (25°N - 5°N )
Nx: int    = 1024                        # Zonal points   
Ny: int    = 512                        # Southern points 



BETA: float = 2.29e-11                  # [s⁻¹ m⁻¹] Rossby parameter
NU_2: float = 0                         # viscosity of order 2  
NU_4: float = 2e13                      # [m⁴/s] hyperviscosity
 

JET_PARAMS: list[tuple] = [
                                       # (lat_center [m], amplitude [m/s], sigma [m])
    ( -3.297*m, -5.166, 1.058*m),                 # Núcleo CLLJ → 15°N real (y=0) 
    (3.719*m, -6.816, 3.946*m),                 # Flanco sur  → 10°N real
   # ( 5*m, -8.0, 7*m),                 # Flanco norte → 20°N rea
    ]
#Latitud 15° <- centro (10°-20°)
#Long 75° <- centro (60°-80°)
# Max V = 15 m/s (Jet core)


WAVE1 = dict(k=4, T_days=4.0, amp=7.5, lat0=-5*m, sigma_y = 5*m, phase0=0.0)
WAVE2 = dict(k=5, T_days=3.5, amp=6.0, lat0=0.0*m, sigma_y=4*m, phase0=np.pi/3)
WAVE3 = dict(k=6, T_days=3.0, amp=3.8, lat0=5*m, sigma_y=4*m, phase0=2*np.pi/3)
WAVE4 = dict(k=7, T_days=5.0, amp=1.5, lat0=3*m, sigma_y=4*m, phase0=5*np.pi/3)

 
NOISE_SIGMA = 0.5                       # [m/s] Synthetic noise
 
# Spectral forcing
NK_MAX_FORCING = 7 #4
NK_MIN_FORCING = 3 #2

# ============================================================================
# SECTION 2 — FLUIDSIM PARAMETER CONFIGURATION
# ============================================================================


params = Simul.create_default_params()
params.oper.type_fft = "fft2d.mpi_with_fftw1d"
# Domain
params.oper.Lx = Lx_deg * m  # [m]
params.oper.Ly = Ly_deg * m  # [m]
params.oper.nx = Nx
params.oper.ny = Ny
params.oper.coef_dealiasing = 2/3 

# Physical parameters
params.beta = BETA
params.nu_2 = NU_2 
params.nu_4 = NU_4

# Temporary integration
params.time_stepping.t_end = period
params.time_stepping.USE_CFL = True
params.time_stepping.deltat_max = float(50)     # [s] 150
params.time_stepping.deltat0 = float(20)        # [s] 100

# Velocity field initialization 
params.init_fields.type = "in_script"

# Activation of the forcing with monkey-patching
params.forcing.enable = ACTIVE_FORCING
params.forcing.type = "in_script"
#params.forcing.nkmax_forcing = NK_MAX_FORCING
#params.forcing.nkmin_forcing = NK_MIN_FORCING
params.forcing.key_forced = "rot_fft"

# Output
params.output.sub_directory                  = "barotropic_cllj_cluster"
params.output.periods_print.print_stdout     = 3600.0       # [s]
params.output.periods_save.phys_fields       = 3600.0       # [s] 
params.output.periods_save.spectra           = 3600.0       # [s] 
params.output.periods_save.spatial_means     = 3600.0       # [s] 
params.output.periods_save.spect_energy_budg = 3600.0       # [s] 
params.output.periods_save.increments        = 3600.0       # [s] 


#------------------------------------------------------------------
# SECTION 3 — PHYSICAL FUNCTIONS
#------------------------------------------------------------------

def Jet_Field(lats: np.ndarray) -> np.ndarray:
    """
    Compute the zonal-mean CLLJ profile as a superposition of Gaussians:

    u_bar = Sum A_i * exp{- frac{(varphi - varphi_i)^2}{2 sigma_i^2}}

    The mean flow is zonally uniform (independent of longitude), so it is
    defined as a 1D function of latitude only.

    Parameters
    ----------
    lats : np.ndarray, shape (ny_local,)
        Latitude coordinate array in metres, centred at the equator (y=0).
 
    Returns
    -------
    u_bar : np.ndarray, shape (ny_local,)
        Zonal velocity profile [m/s]. Negative values = easterly flow.

    """
    u_bar = np.zeros(len(lats))
    for lat0, amp, sigma in JET_PARAMS:
        u_bar += amp * np.exp(-((lats - lat0)**2 / (2* sigma**2)))
    return u_bar

def easterly_wave(
        lats: np.ndarray,
        lons: np.ndarray,
          t: float, 
          wave_params: dict,
          ) ->tuple[np.ndarray, np.ndarray]: 
    """
    Compute the velocity perturbation of a westward-propagating easterly wave.
 
    The wave is represented as a Gaussian envelope in latitude modulated by
    a sinusoidal carrier in longitude that propagates westward:


    u' = A sin(k*lambda + 20t/T)* exp{- frac{(varphi - varphi_0)^2}{2*sigma_y^2}}
    v' = 0.6*A cos(k*lambda + 20t/T)* exp{- frac{(varphi - varphi_0)^2}{2*sigma_y^2}}
    
    Parameters
    ----------
    lats : np.ndarray, shape (ny_local,)   Latitudes [m]
    lons : np.ndarray, shape (nx,)         Longitudes [m]
    t    : float                           Current simulation time [s].
    wave_params : dict
        Keys: k (int), T_days (float), amp (float), lat0 (float), sigma_y (float).
 
    Returns
    -------
    u_prime, v_prime : np.ndarray, shape (ny_local, nx)     Velocity perturbations [m/s].

    """
    
    
    k       = wave_params['k']
    T       = wave_params['T_days'] * 86400.0   # [s]
    amp     = wave_params['amp']
    lat0    = wave_params['lat0']
    sigma_y = wave_params['sigma_y']
    phase   = wave_params['phase0']

    X, Y = np.meshgrid(lons, lats)
    kx = 2*np.pi * k / params.oper.Lx
    omega = 2*np.pi / T
    phi = kx * X + omega * t + phase

    G = np.exp(-((Y - lat0)**2) / (2*sigma_y**2))
    dG_dy = -(Y - lat0) / sigma_y**2 * G

    u_prime = amp * np.sin(phi) * G
    v_prime = amp * 0.6 * np.cos(phi) * G
    
    dvdx    = -0.6 * amp * kx * np.sin(phi) * G
    dudy    =        amp       * np.cos(phi) * dG_dy
    rot     = dvdx - dudy
    return u_prime, v_prime, rot


#______________
#_______________
def add_noise(
    field: np.ndarray,
    sigma: float = NOISE_SIGMA,
    seed: int | None = None,
    ) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to a velocity field.
 
    Noise simulates unresolved mesoscale variability and prevents spectral
    ringing at the grid scale.
 
    Parameters
    ----------
    field : np.ndarray    Field to which noise is added.
    sigma : float         Standard deviation [m/s]. Default: 0.5 m/s.
    seed  : int or None   RNG seed for reproducibility. Default: None (random).
 
    Returns
    -------
    np.ndarray   field + Gaussian noise, same shape as input.
    """
    rng = np.random.default_rng(seed)
    return field + rng.normal(0.0, sigma, field.shape)



#---------------------------------------------------------------------------
# SECTION 4 — SIMULATION INITIALISATION
#
# From this point on, the domain is already broken down into MPI ranks.
# sim.oper.x and sim.oper.y contain ONLY the coordinates of the subdomain
# local of each rank.
#---------------------------------------------------------------------------

sim = Simul(params)
oper = sim.oper 
if rank == 0:
    print("FFT backend =", oper.type_fft)

x = sim.oper.x - params.oper.Lx / 2
y = sim.oper.y - params.oper.Ly / 2

# Definition of the velocity field
u_bar_1d = Jet_Field(y) 
u_mean = np.tile(u_bar_1d[:, None], (1, len(x)))
# Introduction of perturbations
u_prime1, v_prime1, rot1 = easterly_wave(y, x, t=0, wave_params=WAVE1)
u_prime2, v_prime2, rot2 = easterly_wave(y, x, t=0, wave_params=WAVE2)
u_prime3, v_prime3, rot3 = easterly_wave(y, x, t=0, wave_params=WAVE3)
u_prime4, v_prime4, rot4 = easterly_wave(y, x, t=0, wave_params=WAVE4)

U = add_noise(u_mean + u_prime1 + u_prime2 + u_prime3 +u_prime4, seed = rank)
V = add_noise(v_prime1 + v_prime2 + v_prime3 + v_prime4, seed = rank + 1000)

# Vorticity 
dudy = np.gradient(U, oper.deltay, axis=0)
dvdx = np.gradient(V, oper.deltax, axis=1)
rot = dvdx - dudy
omega = oper.fft2(rot)

sim.state.init_from_rotfft(omega)



# ─────────────────────────────────────────────────────────────────────────
# SECTION 5— TIME-DEPENDENT FORCING
#
# Only rank 0 evaluates the function; FluidSim distributes the result.
# oper_coarse is defined outside the if rank==0 so that it is accessible
# within compute_forcingc_each_time from any process.
# ─────────────────────────────────────────────────────────────────────────

if params.forcing.enable:
    forcing_maker = sim.forcing.forcing_maker

    def compute_forcing_each_time(self):
        t_now = sim.time_stepping.t
        _, _, rot1 = easterly_wave(y, x, t_now, WAVE1)
        _, _, rot2 = easterly_wave(y, x, t_now, WAVE2)
        _, _, rot3 = easterly_wave(y, x, t_now, WAVE3)
        _, _, rot4 = easterly_wave(y, x, t_now, WAVE4)
        rot = rot1 + rot2 + rot3 +rot4

        T_forcing = 4 * 86400.0
        rot_norm = rot / T_forcing
        return rot_norm  # [s⁻¹] en espacio físico

    forcing_maker.monkeypatch_compute_forcing_each_time(compute_forcing_each_time)





# ─────────────────────────────────────────────────────────────
# SECTION 6 — RUN
# ─────────────────────────────────────────────────────────────
if rank == 0:
    u_bar_check = Jet_Field(y)
    ke_jet = 0.5 * np.mean(u_bar_check**2)
    print(f"KE inicial del jet: {ke_jet:.3f} J/kg")
    print(f"u_bar min/max: {u_bar_check.min():.2f} / {u_bar_check.max():.2f} m/s")


sim.time_stepping.start()

if rank == 0:
    t_final = time.time()
    t_total = t_final - t_inicio
    print(f"Simulation time: {t_total/60:.2f} minutes")

if rank == 0:
    print(
        "\nTo display a video of this simulation, you can do:\n"
        f"cd {sim.output.path_run}; fluidsim-ipy-load"
        + """

# then in ipython (copy the line in the terminal):

sim.output.phys_fields.animate('b', dt_frame_in_sec=0.1, dt_equations=0.1)
"""
    )


