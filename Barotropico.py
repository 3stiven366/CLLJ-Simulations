from fluidsim.solvers.ns2d.solver import Simul
import numpy as np
from fluiddyn.util.mpi import rank #Necesario para crear el forzamiento y trabajar con MPI

# Parámetros del simulador

params = Simul.create_default_params()

#Tamaño del dominio 

params.oper.Lx = 2 * np.pi
params.oper.Ly = np.pi 
params.oper.nx = nx = 256
params.oper.ny = nx // 2

params.time_stepping.t_end = 8.0
params.time_stepping.deltat0 = 1e-1

#Viscosidad y desaliasing

params.nu_2 = 1e-3
params.oper.coef_dealiasing = 2/3

#Acá se define que el campo de velocidades se tiene que definir en el script
params.init_fields.type = "in_script"


#Activacion del forzamiento con monkey-patching
params.forcing.enable = True
params.forcing.type = "in_script_coarse"
params.forcing.nkmax_forcing = 8 
params.forcing.key_forced = "rot_fft"



params.output.sub_directory = "examples"
params.output.periods_print.print_stdout = 0.5
params.output.periods_save.phys_fields = 0.2
params.output.periods_save.spectra = 0.5
params.output.periods_save.spatial_means = 0.05
params.output.periods_save.spect_energy_budg = 1.0
params.output.periods_save.increments = 1.0


sim = Simul(params)

#_____________________________________________________
#inicialización del campo
#_____________________________________________________
oper = sim.oper
X = sim.oper.X
Y = sim.oper.Y

#jet Zonal: Velocidad solo en x, función de y

U0 = 1.0
y0 = sim.params.oper.Ly / 2
width = 0.2 * sim.params.oper.Ly

ux = U0 * np.tanh((Y-y0) / width)
uy = np.zeros_like(ux)
dudy = np.gradient(ux, oper.deltay, axis=0)

rot = -dudy

rot_fft = oper.fft2(rot)

sim.state.init_from_rotfft(rot_fft)

#______________________________________________________
# Creación del forzamiento
#______________________________________________________
forcing_maker = sim.forcing.forcing_maker

if rank == 0:
    oper = forcing_maker.oper_coarse #Este es un operador reducido
    X, Y = oper.X, oper.Y

    x0, y0 = oper.lx/2, oper.ly/2
    R = oper.lx/10
    A0 = 8.0

    vortex_profile =  A0 * np.exp( -((X-x0)**2 + (Y-y0)**2) / R**2 )

def compute_forcingc_each_time(self):
    if rank != 0:
        return
    t = sim.time_stepping.t

    #Inyección periodica:

    omega = 2 * np.pi / 5.0 # periodo = 5 s-1

    A_t = np.sin(omega * t)

    return A_t * vortex_profile

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







































