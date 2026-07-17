"""
CLLJ_simulation.py
==============================
Two-dimensional barotropic vorticity simulation of Easterly Wave–Caribbean
Low-Level Jet (EW–CLLJ) interactions over the Intra-Americas Seas (IAS).

Physical motivation
-------------------
Rivera (2026) documents that the CLLJ provides a barotropically unstable
mean-flow environment for tropical easterly waves during boreal summer. The
Rayleigh–Kuo criterion (∂q̄/∂y changes sign) is satisfied over most of the
OTREC 2019 period, and positive eddy momentum covariance ⟨u′v′⟩ at
Guanacaste and San Andrés indicates mean-to-eddy energy transfer (CK > 0).

This model captures those mechanisms using the 2D barotropic vorticity
equation on a β-plane:

    ∂ζ/∂t + J(ψ, ζ + f) = ν₄∇⁴ζ + F

where ζ = ∇²ψ is the relative vorticity, f = f₀ + βy is the Coriolis
parameter, ν₄ is the hyperviscosity, and F is the prescribed vorticity
forcing that continuously replenishes the easterly-wave structure.

Diagnostics of interest
-----------------------
Rayleigh–Kuo criterion  : β − ∂²ū/∂y²  (sign change ⇒ necessary condition)
Barotropic conversion   : CK = −⟨u′v′⟩ ∂ū/∂y  (CK > 0 ⇒ jet → eddies)
Eddy kinetic energy     : EKE = ½⟨u′² + v′²⟩

Governing parameters
--------------------
Domain       : 120° × 20°  (100°W–60°W, 5°N–25°N; y = 0 ↔ 15°N)
Resolution   : nx=1024, ny=512  →  Δx ≈ 0.117°, Δy ≈ 0.039°
β            : 2.29×10⁻¹¹ s⁻¹m⁻¹  (tropical β-plane at ~15°N)
ν₄           : 2×10¹³ m⁴/s        (rescaled with resolution as Δx⁴)
Δt_max       : 50 s               (CFL-stable; U_max ≈ 12 m/s)
Integration  : 90 days            (JAS boreal summer season)
Jet profile  : 2-Gaussian fit to ERA5 JAS climatology (925 hPa,
               90°W–80°W, 1991–2020); narrow CLLJ core near 11.7°N
               plus a broad component near 18.7°N
Waves        : 4 modes, k = 4–7, T = 3–5 days (observed EW band)

Forcing formulation
-------------------
The forcing term F is a *rate of vorticity injection* [s⁻²], not a vorticity
field [s⁻¹] — every term in the vorticity equation above carries units of
[s⁻²]. `easterly_wave` returns ζ′ = ∇²ψ′ in [s⁻¹], so each wave mode is
normalised by its own period T_i to obtain the required rate:

    F = Σ_i ζ′_i / T_i          [s⁻¹] / [s] = [s⁻²]

Sustained over a time T_i, the forcing accumulates a vorticity of magnitude
ζ′_i, i.e. it rebuilds one full wave structure in one wave period. This
leaves no free tuning parameter: T_i is fixed by the prescribed wave period.

Passing ζ′ directly as F is dimensionally inconsistent and injects vorticity
orders of magnitude above the mean-flow shear within the first simulated
hour, collapsing the adaptive time step (Δt → 10⁻² s).

The model uses params.forcing.type = "in_script", NOT "in_script_coarse".
With `in_script`, the array returned by compute_forcing_each_time is
transformed by the (normalised) oper.fft and added directly to the nonlinear
tendencies, so the injected amplitude is fully under user control. With
`in_script_coarse`, FluidSim builds a reduced grid whose size it sets
internally and renormalises the forcing, which for a spatially structured
forcing such as a prescribed wave introduces large coarse-to-fine
amplification factors.

MPI notes
---------
This script runs identically with and without MPI:

    python CLLJ_simulation.py                    # serial
    mpirun -np 64 python CLLJ_simulation.py      # parallel

FluidSim decomposes the domain across ranks with the fft2d.mpi_with_fftw1d
backend. Not every (resolution, nproc) pair is valid, and the constraint is
not simply "ny divisible by nproc". The configuration below (ny=512, 64
processes) is verified to run; if the resolution or the process count is
changed, confirm the new pair on a short test run before committing to a
long production job.

Variable scope under MPI
------------------------
Global (identical across all ranks):
    params, period, m, all physical constants, all functions.
Local (each rank holds its own subdomain slice):
    x, y, u_mean, U, V, rot, omega_fft.

Every rank evaluates the forcing on its own local subdomain — x and y already
hold each rank's local coordinates — and FluidSim assembles the global field
internally. No rank-0 guard is used (that pattern belongs to the abandoned
`in_script_coarse` mode, where only rank 0 holds oper_coarse).

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
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the velocity perturbation of a westward-propagating easterly wave,
    derived analytically from a streamfunction.

    The perturbation streamfunction has a Gaussian envelope in latitude
    modulated by a sinusoidal carrier in longitude that propagates westward:

        psi'(x, y, t) = A(y) * sin(theta),   theta = kx*x + omega*t + phase0

        A(y) = amp * sigma_y * sqrt(e) * exp{-(y - y0)^2 / (2*sigma_y^2)}

    Normalization rationale
    ------------------------
    psi' has units of m^2/s, so the prefactor of the Gaussian envelope must
    carry units of m^2/s as well; multiplying the velocity-scale amplitude
    `amp` [m/s] by the length scale `sigma_y` [m] fixes this. The additional
    factor sqrt(e) is chosen so that `amp` equals EXACTLY the peak zonal
    perturbation amplitude max|u'|, reached at y = y0 +/- sigma_y (this is
    a modeling choice made to preserve the amp = max|u'| convention used in
    earlier iterations of this code; it is not drawn from any specific AEW
    reference and should be revisited if a literature-calibrated forcing
    scheme is adopted later).

    Velocity components and vorticity follow by direct differentiation of
    psi', which guarantees non-divergence and dynamical consistency between
    u', v', and zeta' (no ad-hoc amplitude ratio like the previous 0.6
    factor):

        u'    =  d(psi')/dy      = A'(y) * sin(theta)
        v'    = -d(psi')/dx      = -kx * A(y) * cos(theta)
        zeta' =  laplacian(psi') = [A''(y) - kx^2 * A(y)] * sin(theta)

    With this normalization:
        max|u'| = amp                          (exact, at y = y0 +/- sigma_y)
        max|v'| = kx * sigma_y * sqrt(e) * amp  (exact, at y = y0)

    All derivatives are analytical (no finite differences), avoiding the
    spectral noise introduced by np.gradient.

    Parameters
    ----------
    lats : np.ndarray, shape (ny_local,)
        Latitude coordinate array [m].
    lons : np.ndarray, shape (nx,)
        Longitude coordinate array [m].
    t : float
        Current simulation time [s].
    wave_params : dict
        Keys: k (int, zonal wavenumber / number of wave cycles around the
        domain), T_days (float, wave period in days), amp (float, exact peak
        zonal velocity perturbation [m/s]), lat0 (float, latitude of the
        envelope center [m]), sigma_y (float, meridional envelope width [m]),
        phase0 (float, phase offset [rad]).

    Returns
    -------
    u_prime : np.ndarray, shape (ny_local, nx)
        Zonal velocity perturbation [m/s].
    v_prime : np.ndarray, shape (ny_local, nx)
        Meridional velocity perturbation [m/s].
    rot : np.ndarray, shape (ny_local, nx)
        Relative vorticity of the perturbation, zeta' = laplacian(psi') [s⁻¹].
    """
    k       = wave_params['k']
    T       = wave_params['T_days'] * 86400.0   # [s]
    amp     = wave_params['amp']                # [m/s], exact max|u'|
    lat0    = wave_params['lat0']
    sigma_y = wave_params['sigma_y']
    phase   = wave_params['phase0']

    X, Y = np.meshgrid(lons, lats)
    kx = 2 * np.pi * k / params.oper.Lx      # physical zonal wavenumber [rad/m]
    omega = 2 * np.pi / T
    theta = kx * X + omega * t + phase

    dY = Y - lat0
    envelope = np.exp(-(dY**2) / (2 * sigma_y**2))
    C   = amp * sigma_y * np.sqrt(np.e)
    A   = C * envelope                                           # [m^2/s]
    dA  = -(dY / sigma_y**2) * A                                  # A'(y)  [m/s]
    d2A = (dY**2 / sigma_y**4 - 1.0 / sigma_y**2) * A             # A''(y) [1/s]

    u_prime = dA * np.sin(theta)
    v_prime = -kx * A * np.cos(theta)
    rot     = (d2A - kx**2 * A) * np.sin(theta)

    return u_prime, v_prime, rot


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

    def compute_forcing_each_time(self) -> np.ndarray:
        """
        Evaluate the vorticity forcing F at the current time step.
    
        This method is monkey-patched into FluidSim's `in_script` forcing maker and
        is called at every time step. The array it returns is transformed to Fourier
        space by FluidSim (`oper.fft`, which is normalised) and added directly to the
        nonlinear tendencies in `tendencies_nonlin`:
    
            tendencies_fft += self.forcing.get_forcing()
    
        Units
        -----
        `tendencies_fft` represents dzeta/dt, so the returned array MUST have units
        of [s^-2] — it is a *rate of vorticity injection*, not a vorticity field.
        The barotropic vorticity equation
    
            dzeta/dt + J(psi, zeta + f) = nu_4 * lap^2(zeta) + F
    
        is dimensionally consistent only if every term is [s^-2]; this is a property
        of the equation itself, not a FluidSim convention.
    
        `easterly_wave` returns zeta' = lap(psi'), which is a vorticity in [s^-1].
        Dividing by a characteristic time T converts it to the required rate:
    
            F = zeta' / T          [s^-1] / [s] = [s^-2]
    
        Passing zeta' directly (without dividing by T) is dimensionally wrong and
        numerically injects vorticity ~10^2 times larger than the mean jet's within
        a single simulated hour, collapsing the adaptive time step.
    
        Choice of T
        -----------
        Each wave mode is normalised by ITS OWN period T_i rather than by a single
        global constant. The interpretation is that the forcing rebuilds one full
        wave structure over one wave period: sustained for a time T_i, the injected
        vorticity accumulates to zeta'_i. This leaves no free tuning parameter —
        T_i is already fixed by the prescribed wave period, taken from the observed
        2.5-6 day easterly-wave band.
    
            F = sum_i ( zeta'_i / T_i )
    
        MPI behaviour
        -------------
        Every rank evaluates the forcing on its own local subdomain: `x` and `y`
        already hold each rank's local coordinates, so no rank-0 guard is needed.
        FluidSim assembles the global field internally. (This differs from the
        `in_script_coarse` mode, where only rank 0 computes the coarse field.)
    
        Returns
        -------
        F : np.ndarray, shape (ny_local, nx)
            Vorticity injection rate [s^-2] in physical space on the local subdomain.
        """
        t_now = sim.time_stepping.t
    
        F = np.zeros((len(y), len(x)))
    
        for wave_params in (WAVE1, WAVE2, WAVE3, WAVE4):
            _, _, rot = easterly_wave(y, x, t_now, wave_params)
            T_wave = wave_params['T_days'] * 86400.0        # [s]
            F += rot / T_wave                               # [s^-1] / [s] = [s^-2]
    
        return F


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


