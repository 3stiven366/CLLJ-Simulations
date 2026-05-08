from fluidsim.solvers.ns2d.solver import Simul
from fluiddyn.util.mpi import rank
import numpy as np
import time 

t_inicio = time.time()
#------------------------------------------------------------------
# Global Constants
#------------------------------------------------------------------
np.random.seed(42)
days = 10 #días de simulación
days_sec = 86400 * days


params = Simul.create_default_params()

#Dominio 
m = 111e3 #Conversión de grados a metros
Lx = params.oper.Lx = 360 * m #Lons
Ly = params.oper.Ly = 60 * m #Lats
nx = params.oper.nx = 360
ny = params.oper.ny = 180

"""
Valores nu4:
 -- Estudio limpio de inestabilidad (sin disipación en escalas de jet): nu4 = 1e13
 -- Simulación con disipación realista (más cercano a atmósfera real): nu4 =  5e13

 Valores beta:
 -- beta real: beta = 2.29e-11
 -- beta=0 (plano-f, caso límite, inestabilidad máxima): beta =  0.0
 -- beta amplificado (inhibe la inestabilidad): beta = 5e-11
"""
#Parámetros físicos
params.beta = 2.29e-11 #Coriolis
params.nu_2 = 0 
params.nu_4 = 1e13 #viscosidad

#Duración de la simulación
params.time_stepping.t_end = days_sec
params.time_stepping.deltat_max = float(120) #segundos (anteriormente 400s), Criterio de Courant-Friedrichs-Lewy (CFL) para calcular dt
params.time_stepping.deltat0 = float(60) #Valor usado en mi simulación
params.oper.coef_dealiasing = 2/3 # Para subir la resolución efectiva usar 1/2

# Indica que se inicia el campo de velocidad en el código
params.init_fields.type = "in_script"

#Activación del forzamiento con monkey-patching
params.forcing.enable = True
params.forcing.type = "in_script_coarse"
params.forcing.nkmax_forcing = 6#Para grandes dominios se debe usar nk_max = 4--3
params.forcing.nkmin_forcing = 2
params.forcing.key_forced = "rot_fft"

#Parámetros de guardado
params.output.sub_directory = "examples"
params.output.periods_print.print_stdout = 10.0 #0.5
params.output.periods_save.phys_fields = 0.2
params.output.periods_save.spectra = 0.5
params.output.periods_save.spatial_means = 0.05
params.output.periods_save.spect_energy_budg = 2.0
params.output.periods_save.increments = 1.0

#------------------------------------------------------------------
# Defining Functions
#------------------------------------------------------------------

def Jet_Field(lats):
    """
    This function creates an array, u_bar, wich represents 
    the jet field. This field is defined as:

    u_bar = Sum A_i * exp{- frac{(varphi - varphi_i)^2}{2 sigma_i^2}}

    The eastern jet stream is constructed as a superposition 
    of three Gaussian lines in latitude, simulating a zonally
    uniform and climatologically realistic profile.
    """
    u_bar = np.zeros(len(lats))
    for lat, amp, sigma in [
        (-10*m, -0.6, 8*m),# núcleo principal
        (10*m, -0.8, 7*m), # flanco ecuatorial (zona inestable)
        (0, -0.3, 15*m),  # flanco polar
        ]: # [amp] = m/s
        u_bar += amp * np.exp(-((lats - lat)**2 / (2* sigma**2)))
    return u_bar

def easterly_wave(lats, lons, t, k=6, period=3.5, amp=0.3, lat0=10*m, sigma_y=20*m): #periodo de las ondas: 3.5 días
    """
    Easterly waves are created as a Gaussian perturbation
    moving to the west. They are representing as:

    u' = A sin(k*lambda + 20t/T)* exp{- frac{(varphi - varphi_0)^2}{2*sigma_y^2}}
    v' = 0.6*A cos(k*lambda + 20t/T)* exp{- frac{(varphi - varphi_0)^2}{2*sigma_y^2}}
    """
    X, Y = np.meshgrid(lons, lats)
    kx = 2*np.pi * k / params.oper.Lx
    omega = 2*np.pi / (period * 86400)
    phi = kx * X + omega * t # fase → oeste  
    envelope = np.exp(-((Y - lat0)**2) / (2*sigma_y**2))
    u_prime = amp * np.sin(phi) * envelope
    v_prime = amp * 0.6 * np.cos(phi) * envelope
    return u_prime, v_prime

def add_noise(field, scale=0.5):
    """
    Function that adds noise sigma = 0.5 m/s
    to simulate mesoscale variability and avoid 
    filtering artifacts. Noise does not systematically 
    contribute to CBT (its mean is zero), but it does 
    enrich the power spectrum.
    """
    return field + np.random.normal(0, scale, field.shape)


#------------------------------------------------------------------
# Execution
#------------------------------------------------------------------
sim = Simul(params)

oper = sim.oper 

x = sim.oper.x
y = sim.oper.y - params.oper.Ly/2

#Definición del campo
u_bar_1d = Jet_Field(y)
u_mean = np.tile(u_bar_1d[:, None], (1, nx)) #Esto crea una grilla de velocidades 2D 

#Introducción de las perturbaciones 
u_prime, v_prime = easterly_wave(y, x, t=0)

#Campo total
U = add_noise(u_mean + u_prime, scale=0.05)   # 5 cm/s — solo siembra modos
V = add_noise(v_prime, scale=0.05)

#Vorticidad 
dudy = np.gradient(U, oper.deltay, axis=0)
dvdx = np.gradient(V, oper.deltax, axis=1)
rot = dvdx - dudy
omega = oper.fft2(rot)

sim.state.init_from_rotfft(omega)



# ─────────────────────────────────────────────────────────────
# FORZAMIENTO TEMPORAL
# Solo rank 0 evalúa la función; FluidSim distribuye el resultado.
# oper_coarse se define fuera del if rank==0 para que sea accesible
# dentro de compute_forcingc_each_time desde cualquier proceso.
# En una corrida sin MPI, rank == 0 siempre, asi que el comportamiento
# es identico al original.
# ─────────────────────────────────────────────────────────────

forcing_maker = sim.forcing.forcing_maker

if rank == 0:
    # Estas variables SOLO EXISTEN en rank 0
    oper_coarse = forcing_maker.oper_coarse
    x_c = oper_coarse.x
    y_c = oper_coarse.y - params.oper.Ly / 2


def compute_forcingc_each_time(self):
    """
    Computes the time-dependent forcing at each timestep:

    - Only runs on the master process (rank 0).
    - Uses current simulation time to evaluate the easterly wave.
    - Builds velocity perturbations (u_f, v_f) on the coarse grid.
    - Converts them into vorticity forcing: rot_f = dv/dx - du/dy.
    - Returns the vorticity forcing, which is added to the evolution equation.

    This injects energy continuously into the flow with a prescribed
    spatial structure and temporal evolution.
    """
    if rank != 0:
        return None     

    u_f, v_f = easterly_wave(y_c, x_c, t = sim.time_stepping.t)

    # convertir a vorticidad
    dudy = np.gradient(u_f, oper_coarse.deltay, axis=0)
    dvdx = np.gradient(v_f, oper_coarse.deltax, axis=1)

    rot_f = dvdx - dudy

    return rot_f 


forcing_maker.monkeypatch_compute_forcingc_each_time(compute_forcingc_each_time)

# ─────────────────────────────────────────────────────────────
# INTEGRACIÓN TEMPORAL
# ─────────────────────────────────────────────────────────────

sim.time_stepping.start()

if rank == 0:
    t_final = time.time()
    t_total = t_final - t_inicio
    print(f"Tiempo de simulación: {t_total/60:.2f} minutos")

if rank == 0:
    print(
        "\nTo display a video of this simulation, you can do:\n"
        f"cd {sim.output.path_run}; fluidsim-ipy-load"
        + """

# then in ipython (copy the line in the terminal):

sim.output.phys_fields.animate('b', dt_frame_in_sec=0.1, dt_equations=0.1)
"""
    )


