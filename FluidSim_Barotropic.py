from fluidsim.solvers.ns2d.solver import Simul
from fluiddyn.util.mpi import rank
import numpy as np

#------------------------------------------------------------------
# Global Constants
#------------------------------------------------------------------

params = Simul.create_default_params()

#Dominio 
m = 111e3 #Conversión de grados a metros
Lx = params.oper.Lx = 360 * m #Lons
Ly = params.oper.Ly = 60 * m #Lats
nx = params.oper.nx = 120
ny = params.oper.ny = 60



params.beta = 2.29e-11
params.nu_2 = 1e-3 
params.nu_4 = 1e14
params.time_stepping.deltat_max = float(400) #segundos --> sugerido 1h 
params.time_stepping.deltat0 = float(200) #Valor usado en mi simulación
params.oper.coef_dealiasing = 2/3 #¡¡¡Fijarse si es necesario!!!!

# Indica que se inicia el campo de velocidad en el código
params.init_fields.type = "in_script"


# Cambiar el forzamiento

#Activación del forzamiento con monkey-patching
params.forcing.enable = True
params.forcing.type = "in_script_coarse"
params.forcing.nkmax_forcing = 6#Para grandes dominios se debe usar nk_max = 4--3
params.forcing.nkmin_forcing = 2



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
    for lat, amp, sigma in [(-10*m, -0.6, 8*m), (10*m, -0.8, 7*m), (0, -0.3, 15*m)]: # [amp] = m/s
        u_bar += amp * np.exp(-((lats - lat)**2 / (2* sigma**2)))
    return u_bar

def easterly_wave(lats, lons, t, k=6, period=5, amp=0.3, lat0=10*m, sigma_y=20*m):
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
# Las otra perturbación está faltando, probar primero con la primera perturbación


#Campo total
U = u_mean + u_prime
V = v_prime

dudy = np.gradient(U, oper.deltay, axis=0)
dvdx = np.gradient(V, oper.deltax, axis=1)
rot = dvdx - dudy
omega = oper.fft2(rot)

sim.state.init_from_rotfft(omega)


#Forzamiento con evolución temporal

forcing_maker = sim.forcing.forcing_maker


if rank == 0:
    oper_coarse = forcing_maker.oper_coarse
    x_c = oper_coarse.x
    y_c = oper_coarse.y - params.oper.Ly/2

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
        return

    u_f, v_f = easterly_wave(y_c, x_c, t = sim.time_stepping.t)

    # convertir a vorticidad
    dudy = np.gradient(u_f, oper_coarse.deltay, axis=0)
    dvdx = np.gradient(v_f, oper_coarse.deltax, axis=1)

    rot_f = dvdx - dudy
    forcing_strength = 1e-3  # Necesario para ensuavisar delta energía

    return forcing_strength * rot_f




forcing_maker.monkeypatch_compute_forcingc_each_time(compute_forcingc_each_time)


sim.time_stepping.start()
if rank == 0:
    print(
        "\nTo display a video of this simulation, you can do:\n"
        f"cd {sim.output.path_run}; fluidsim-ipy-load"
        + """

# then in ipython (copy the line in the terminal):

sim.output.phys_fields.animate('b', dt_frame_in_sec=0.1, dt_equations=0.1)
"""
    )