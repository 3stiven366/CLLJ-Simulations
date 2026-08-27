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
from fluiddyn.util.mpi import rank, comm, nb_proc  
import numpy as np
import time 

t_inicio = time.time()
# ============================================================================
#  SECTION 1 — GLOBAL CONSTANTS 
# ============================================================================

ACTIVE_FORCING = True                   # True / False

np.random.seed(42)
m: float = 111e3                        # Degrees to meters
days: int = 2                          # Days of simulation
period: int = 86400 * days              # [s]

Lx_deg: int = 40                       # [°]  zonal domain (100°W - 60°W)
Ly_deg: int = 20                        # [°] southern domain (25°N - 5°N )
Nx: int    = 256 #1024                        # Zonal points   
Ny: int    = 128 #512                        # Southern points 

TAU_RELAX: float = 7 * 86400.0          # [s] escala de relajación del jet (~7 días)
BETA: float = 2.29e-11                  # [s⁻¹ m⁻¹] Rossby parameter

#________________________
# ── Viscosidad de eddy (orden 2): fija la vida física de las ondas ──────────
# nu_2 = L_R^2 / tau_spindown,  con L_R = sqrt(c/beta)  =>  nu_2 = c/(beta*tau)
# c: velocidad de fase representativa de la banda AEW; tau: ~10 días (literatura)
C_REPRESENTATIVE: float = 9.0                      # [m/s] centro de la banda AEW
TAU_SPINDOWN: float     = 10 * 86400.0             # [s]
NU_2: float             = C_REPRESENTATIVE / (BETA * TAU_SPINDOWN)   # ≈ 4.5e5 m²/s   (viscosity of order 2)
NU_4: float             = 2e13                                       # [m⁴/s] hyperviscosity
#_________________________


JET_PARAMS: list[tuple] = [  
    # (lat_center [m], amplitude [m/s], sigma [m])
    ( -3.297, -5.166, 1.058),          
    (3.719, -6.816, 3.946),            
     ( 5, -8.0, 7),                 
    ]



# ── Catálogo de eventos de onda del este ────────────────────────────────────
CATALOG_SEED       = 12345
EVENT_SPACING_DAYS = 7.0     # tiempo entre nacimientos consecutivos [días]
T_INJECT_DAYS      = 3.0     # duración de la inyección (canilla abierta) [días]
T_RAMP_DAYS        = 1.0     # rampas suaves de subida/bajada [días]

WAVE_RANGES = dict(
    wavelength = (2.5e6, 4.0e6),    # [m]   banda objetivo (se cuantiza, ver nota)
    c_phase    = (7.0, 12.0),       # [m/s] fase hacia el oeste
    amp        = (3.0, 8.0),        # [m/s] pico de v'
    lat0_deg   = (11.0, 17.0),      # [°N]
    sigma_y    = (2.5, 5.0),        # [°]
    sigma_x    = (4.0, 7.0),        # [°]
)

# NOTE: the 40°-wide periodic domain only admits integer numbers of wave cycles,
# so wavelengths are quantized to Lx/n (n=1,2 here → 4440, 2220 km). The AEW band
# 2500–4000 km is therefore represented by two discrete values only. Whether a
# domain this narrow is adequate for the target region is left as an open question
# to review with the advisor; the quantization keeps the carrier periodic meanwhile.

#________________________

CATALOG_SEED = 12345
GAP_DAYS     = 0.0
 
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
        lat0 = (lat0 - 5.0) * m
        sigma = sigma * m
        u_bar += amp * np.exp(-((lats - lat0)**2 / (2* sigma**2)))
    return u_bar

def jet_vorticity(lats: np.ndarray) -> np.ndarray:
    """Vorticidad analítica del jet base: zeta_jet = -du_bar/dy."""
    zeta = np.zeros(len(lats))
    for lat0, amp, sigma in JET_PARAMS:
        lat0 = (lat0 - 5.0) * m
        sigma = sigma * m
        dudy = amp * np.exp(-((lats - lat0)**2) / (2*sigma**2)) * (-(lats - lat0)/sigma**2)
        zeta += -dudy
    return zeta

def quantize_wavelength(lam_target, Lx):
    """
    Snap a desired wavelength to one that fits an INTEGER number of full cycles
    in the periodic domain: lambda_q = Lx / n, with n = round(Lx / lam_target).

    Why this is necessary
    ---------------------
    The carrier sin(k x + phase) is only continuous across the periodic boundary
    x=0 ≡ x=Lx when k*Lx = 2*pi*n with n integer, i.e. when a whole number of
    wavelengths fits in the domain. Any other wavelength leaves a jump at the seam
    that the FFT turns into Gibbs ringing (spurious energy across all scales,
    injected every time step). Rounding n to the nearest integer removes the jump.

    n is floored at 1 (n=0 would be a spatially constant field, i.e. no wave).
    Note: on a 40° domain only n=1,2 fall in the AEW band, so distinct target
    wavelengths can collapse onto the same quantized value.
    """
    n = max(1, round(Lx / lam_target))
    return Lx / n, n


def build_wave_catalog(t_max, Lx, seed=CATALOG_SEED, verbose=True):
    """
    Sequential (non-overlapping) train of easterly-wave events. Deterministic:
    identical on every MPI rank because it uses a fixed seed and no MPI state.

    Each event injects during a short window (T_INJECT_DAYS) shaped by a smooth
    Hann-like envelope, then switches off, leaving a long free-evolution window
    in which the already-injected wave crosses the jet and decays under nu_2.
    """
    rng = np.random.default_rng(seed)
    events, t = [], 0.0
    spacing = EVENT_SPACING_DAYS * 86400.0
    while t < t_max:
        lam_target = rng.uniform(*WAVE_RANGES['wavelength'])
        lam_q, n   = quantize_wavelength(lam_target, Lx)
        c_p        = rng.uniform(*WAVE_RANGES['c_phase'])
        events.append(dict(
            event_id   = len(events),
            t_birth    = t,
            wavelength_target = lam_target,     # sorteado (para el reporte)
            wavelength = lam_q,                 # cuantizado (el que se usa)
            n_cycles   = n,
            c_phase    = c_p,
            T          = lam_q / c_p,           # período coherente con λ cuantizada
            x_entry    = rng.uniform(0.0, Lx),  # posición inicial del centro
            amp        = rng.uniform(*WAVE_RANGES['amp']),
            lat0       = (rng.uniform(*WAVE_RANGES['lat0_deg']) - 5.0) * m,
            sigma_y    = rng.uniform(*WAVE_RANGES['sigma_y']) * m,
            sigma_x    = rng.uniform(*WAVE_RANGES['sigma_x']) * m,
            phase      = rng.uniform(0.0, 2.0 * np.pi),
            t_inject   = T_INJECT_DAYS * 86400.0,
            t_ramp     = T_RAMP_DAYS   * 86400.0,
        ))
        t += spacing

    if verbose and rank == 0:
        print(f"\n{'='*66}\nCatálogo de ondas: {len(events)} eventos "
              f"(dominio {Lx/m:.0f}°, λ cuantizada a Lx/n)")
        print(f"{'id':>3} {'nace(d)':>8} {'λ_sorteada':>11} {'n':>2} "
              f"{'λ_real':>8} {'cambio':>8} {'c(m/s)':>7}")
        for ev in events:
            lt, lq = ev['wavelength_target'], ev['wavelength']
            print(f"{ev['event_id']:>3} {ev['t_birth']/86400:>8.1f} "
                  f"{lt/1e3:>9.0f}km {ev['n_cycles']:>2} {lq/1e3:>6.0f}km "
                  f"{(lq-lt)/lt*100:>+7.1f}% {ev['c_phase']:>7.1f}")
        print(f"{'='*66}\n")

    return events


def wave_envelope(t, ev):
    """
    Smooth temporal envelope W(t) ∈ [0,1]: rises 0→1 (cosine ramp), holds, then
    falls 1→0 over the injection window [t_birth, t_birth+t_inject]. Zero afterward.
    Continuous dW/dt at both ends so the adaptive time step is never kicked.
    """
    s  = t - ev['t_birth']
    ti = ev['t_inject']
    tr = ev['t_ramp']
    if s < 0.0 or s > ti:
        return 0.0
    if s < tr:
        return 0.5 * (1.0 - np.cos(np.pi * s / tr))
    if s > ti - tr:
        return 0.5 * (1.0 - np.cos(np.pi * (ti - s) / tr))
    return 1.0

def easterly_wave_event(lats, lons, t, ev):
    """
    zeta' [s^-1] of one easterly-wave event at time t, times the temporal
    envelope W(t). Two changes vs the earlier version:

      1. Periodic zonal distance: delta_x = (X - x_c + Lx/2) % Lx - Lx/2.
         With the quantized wavelength (integer cycles in Lx) this keeps BOTH
         the Gaussian envelope and the carrier continuous across the seam.
      2. Temporal envelope W(t): short injection, then free evolution.

    x_c(t) = x_entry - c_phase*(t - t_birth), wrapped into the domain by the
    periodic distance. The packet propagates westward and, being on a periodic
    ring, re-enters smoothly rather than dying at a boundary; its lifetime is
    set by nu_2 dissipation during the free-evolution window, not by leaving
    the domain.
    """
    W = wave_envelope(t, ev)
    if W == 0.0:
        return np.zeros((len(lats), len(lons)))

    Lx = Lx_deg * m
    sigma_x, sigma_y = ev['sigma_x'], ev['sigma_y']
    k     = 2.0 * np.pi / ev['wavelength']
    omega = 2.0 * np.pi / ev['T']
    tau   = t - ev['t_birth']
    x_c   = ev['x_entry'] - ev['c_phase'] * tau

    X, Y = np.meshgrid(lons, lats)
    delta_x = (X - x_c + Lx / 2.0) % Lx - Lx / 2.0   # periodic zonal distance
    delta_y = Y - ev['lat0']
    theta   = k * X + omega * tau + ev['phase']

    G   = np.exp(-delta_y**2 / (2*sigma_y**2)) * np.exp(-delta_x**2 / (2*sigma_x**2))
    C   = ev['amp'] / k
    Psi = C * G * np.sin(theta)

    curvature = (
        sigma_x**4 * delta_y**2
        + sigma_y**4 * delta_x**2
        - sigma_x**4 * sigma_y**4 * k**2
        - sigma_x**2 * sigma_y**4
        - sigma_x**4 * sigma_y**2)

    rot = (-curvature * Psi
           - 2 * C * G * sigma_x**2 * sigma_y**4 * k * delta_x * np.cos(theta)
           ) / (sigma_x**4 * sigma_y**4)

    return rot * W


def active_event(t_now, catalog=None):
    """Devuelve el evento activo en t_now, o None si estamos en un gap."""
    if catalog is None:
        catalog = WAVE_CATALOG
    for ev in catalog:
        if ev['t_birth'] <= t_now <= ev['t_death']:
            return ev
    return None


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

WAVE_CATALOG = build_wave_catalog(period, Lx_deg * m)
if rank == 0:
    print(f"Catálogo: {len(WAVE_CATALOG)} eventos de onda")

sim = Simul(params)
oper = sim.oper 
if rank == 0:
    print("FFT backend =", oper.type_fft)

x = sim.oper.x 
y = sim.oper.y 

# Definition of the velocity field
u_bar_1d = Jet_Field(y) 
u_mean = np.tile(u_bar_1d[:, None], (1, len(x)))

U = add_noise(u_mean, seed = rank)
V = np.zeros_like(U)  # No mean meridional flow

# Vorticity 
dudy = np.gradient(U, oper.deltay, axis=0)
dvdx = np.gradient(V, oper.deltax, axis=1)
rot = dvdx - dudy
omega = oper.fft2(rot)

sim.state.init_from_rotfft(omega)


ZETA_JET_TARGET = jet_vorticity(y)
DENOM = float(np.sum(ZETA_JET_TARGET**2))

if nb_proc > 1:
    DENOM = comm.allreduce(DENOM)


# ─────────────────────────────────────────────────────────────────────────
# SECTION 5 — DIAGNOSTIC LOGGING
# ─────────────────────────────────────────────────────────────────────────
import os
import csv

LOG_EVERY_S   = 600.0      # cadencia de registro de series temporales [s]
FLUSH_EVERY_N = 200        # filas en buffer antes de volcar a disco

_diag_rows  = []
_last_log_t = [-np.inf]    # lista de 1 elemento: mutable desde el closure
_diag_path  = [None]


def global_mean(field_local):
    """Media global de un campo distribuido en y (slab decomposition)."""
    s = float(field_local.sum())
    n = int(field_local.size)
    if nb_proc > 1:
        s = comm.allreduce(s)
        n = comm.allreduce(n)
    return s / n


def write_wave_catalog(path_run):
    """Vuelca el catálogo completo de eventos a CSV. Solo rank 0, una vez."""
    if rank != 0:
        return
    p = os.path.join(path_run, "wave_catalog.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "event_id", "t_birth_days", "t_death_days", "duration_days",
            "wavelength_km", "T_days", "c_phase_ms", "c_group_ms",
            "amp_ms", "lat0_deg", "sigma_y_deg", "sigma_x_deg",
            "phase_rad", "x_entry_km",
        ])
        for ev in WAVE_CATALOG:
            w.writerow([
                ev['event_id'],
                round(ev['t_birth'] / 86400.0, 4),
                round(ev['t_death'] / 86400.0, 4),
                round((ev['t_death'] - ev['t_birth']) / 86400.0, 4),
                round(ev['wavelength'] / 1e3, 2),
                round(ev['T'] / 86400.0, 4),
                round(ev['wavelength'] / ev['T'], 4),
                round(ev['c_group'], 4),
                round(ev['amp'], 4),
                round(ev['lat0'] / m + 5.0, 4),
                round(ev['sigma_y'] / m, 4),
                round(ev['sigma_x'] / m, 4),
                round(ev['phase'], 6),
                round(ev['x_entry'] / 1e3, 2),
            ])
    print(f"[rank 0] Catálogo escrito: {p} ({len(WAVE_CATALOG)} eventos)")


def init_diagnostics(path_run):
    """Crea el CSV de series temporales con su encabezado. Solo rank 0."""
    if rank != 0:
        return
    _diag_path[0] = os.path.join(path_run, "jet_diagnostics.csv")
    with open(_diag_path[0], "w", newline="") as f:
        csv.writer(f).writerow([
            "t_days", "A_jet", "event_id", "KE_zonal", "KE_eddy", "F_jet_rate",
        ])


def flush_diagnostics():
    """Vuelca el buffer de filas al CSV. Solo rank 0."""
    if rank != 0 or not _diag_rows:
        return
    with open(_diag_path[0], "a", newline="") as f:
        csv.writer(f).writerows(_diag_rows)
    _diag_rows.clear()


def split_kinetic_energy():
    """Separa la EC en flujo medio zonal vs eddies. Colectivo (todos los ranks)."""
    ux = sim.state.get_var('ux')
    uy = sim.state.get_var('uy')
    ux_bar = ux.mean(axis=1, keepdims=True)
    uy_bar = uy.mean(axis=1, keepdims=True)
    ke_zon = 0.5 * global_mean(np.broadcast_to(ux_bar**2 + uy_bar**2, ux.shape))
    ke_edd = 0.5 * global_mean((ux - ux_bar)**2 + (uy - uy_bar)**2)
    return ke_zon, ke_edd


# ─────────────────────────────────────────────────────────────────────────
# SECTION 6— TIME-DEPENDENT FORCING
#
# Only rank 0 evaluates the function; FluidSim distributes the result.
# oper_coarse is defined outside the if rank==0 so that it is accessible
# within compute_forcingc_each_time from any process.
# ─────────────────────────────────────────────────────────────────────────



if params.forcing.enable:
    forcing_maker = sim.forcing.forcing_maker

    def compute_forcing_each_time(self) -> np.ndarray:
        t_now = sim.time_stepping.t
        F = np.zeros((len(y), len(x)))

        # --- Onda del este activa (a lo sumo una) ---
        ev = active_event(t_now)
        event_id = ev['event_id'] if ev is not None else -1
        if ev is not None:
            F += easterly_wave_event(y, x, t_now, ev) / ev['T']

        # --- Nudging unidireccional del jet (solo piso) ---
        rot_now  = sim.oper.ifft2(sim.state.state_spect.get_var('rot_fft'))
        zeta_bar = rot_now.mean(axis=1)

        num = float(np.sum(zeta_bar * ZETA_JET_TARGET))
        if nb_proc > 1:
            num = comm.allreduce(num)
        A = num / DENOM

        F_rate = 0.0
        if A < 1.0:
            F_rate = (1.0 - A) / TAU_RELAX
            F += F_rate * ZETA_JET_TARGET[:, None]

        # --- Registro de diagnósticos ---
        if t_now - _last_log_t[0] >= LOG_EVERY_S:
            _last_log_t[0] = t_now
            ke_z, ke_e = split_kinetic_energy()      # colectivo: TODOS los ranks
            if rank == 0:
                _diag_rows.append([
                    round(t_now / 86400.0, 6), round(A, 8), event_id,
                    round(ke_z, 8), round(ke_e, 8), F_rate,
                ])
                if len(_diag_rows) >= FLUSH_EVERY_N:
                    flush_diagnostics()

        return F


    forcing_maker.monkeypatch_compute_forcing_each_time(compute_forcing_each_time)



# ─────────────────────────────────────────────────────────────
# SECTION 7 — RUN
# ─────────────────────────────────────────────────────────────
if rank == 0:
    u_bar_check = Jet_Field(y)
    ke_jet = 0.5 * np.mean(u_bar_check**2)
    print(f"KE inicial del jet: {ke_jet:.3f} J/kg")
    print(f"u_bar min/max: {u_bar_check.min():.2f} / {u_bar_check.max():.2f} m/s")



write_wave_catalog(sim.output.path_run)
init_diagnostics(sim.output.path_run)

sim.time_stepping.start()

flush_diagnostics()      # volcar lo que quede en buffer al terminar


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


