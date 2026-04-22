from fluidsim.solvers.ns2d.solver import Simul
from fluiddyn.util.mpi import rank
import numpy as np

params = Simul.create_default_params()

params.oper.Lx = 2 * np.pi * 6371e3 * np.cos(0)
params.oper.Ly = 60 * np.pi / 180 * 6371e3
params.oper.nx = 640#1024 "Valores de resolución recomendandos para el cluster"
params.oper.ny = 380 #512

#params.f0 = 0.0 #Grok lo sugiere, pero este valor no existe al menos con el método params.
params.beta = 2.29e-11
params.nu_2 = 1e-3 
params.nu_4 = 1e14
params.time_stepping.deltat_max = 40.0 #segundos --> sugerido 1h 
params.time_stepping.deltat0 = 1e-1 #Valor usado en mi simulación
params.oper.coef_dealiasing = 2/3 #¡¡¡Fijarse si es necesario!!!!


# Indica quese inicia el campo de velocidad en el código
params.init_fields.type = "in_script"

#Activación del forzamiento con monkey-patching
params.forcing.enable = True
params.forcing.type = "random"
params.forcing.nkmax_forcing = 6#Para grandes dominios se debe usar nk_max = 4--3
params.forcing.nkmin_forcing = 2
params.forcing.random._set_attrib("time_correlation", 6 * 3600)
# Inyección débil

params.forcing.forcing_rate = 1e-6



#__________________________________________________
#Explicación del forzamiento:
#_________________________________________________

"""
Este forzamiento se utiliza porque permite **inyectar energía 
únicamente en las ondas**, sin imponer directamente la estructura 
ni la intensidad del flujo zonal.
Al ser **estocástico**, no introduce una fase ni una frecuencia
preferencial, lo que evita forzar artificialmente una onda específica
y permite analizar el intercambio de energía de manera 
**estadística**, como ocurre en la atmósfera real con procesos 
convectivos no resueltos. El hecho de que el forzamiento actúe 
solo en **números de onda bajos** garantiza que la energía 
se inyecte en **escalas largas tropicales**, que son las relevantes
para la interacción con jets barotrópicamente inestables. 
Además, la **tasa de inyección débil** asegura que la dinámica
esté dominada por la interacción no lineal entre ondas y flujo medio 
y no por la fuente externa. Finalmente, al **excluir el modo zonal** 
((k_x=0)), el flujo medio puede evolucionar libremente, de modo que 
cualquier ganancia o pérdida de energía del jet proviene 
exclusivamente de la transferencia con las ondas, permitiendo 
identificar de forma clara y físicamente consistente el 
_lintercambio de energía ondas–corriente media.

"""


#_____________________________________________

params.output.sub_directory = "examples"
params.output.periods_print.print_stdout = 0.5
params.output.periods_save.phys_fields = 3600 #cada hora
params.output.periods_save.spectra = 0.5
params.output.periods_save.spatial_means =600  # 0.05 #10 min
params.output.periods_save.spect_energy_budg = 1.0
params.output.periods_save.increments = 3600 #1.0


sim = Simul(params)

#_____________________________________________________
#inicialización del campo
#_____________________________________________________
oper = sim.oper
X = sim.oper.X #A diferencia de lo que Grok ofrece, estos permiten MPI y es más seguro
Y = sim.oper.Y


#Flujo Zonal Inestable: 

U0 = 25.0
lat0 = 0.0
sigma = 8.0 * np.pi/180 * 6371e3

u_mean = U0 * np.exp( -((Y - lat0*6371e3*np.pi/180)**2) / (2*sigma**2))
v_mean = np.zeros_like(u_mean)

k = 3
A = 0.5 
v_pert = A * np.cos(k * X / (6371e3 * 2*np.pi / 360)) * np.exp(-((Y)/ (10*sigma))**2)

dudy = np.gradient(u_mean, oper.deltay, axis=0)
dvdx = np.gradient(v_mean + v_pert, oper.deltax, axis=0)
rot = dvdx - dudy
omega = oper.fft2(rot)

sim.state.init_from_rotfft(omega)

#params.time_stepping.t_end = 30 * 86400   # 30 días (acá se puede configurar la cantidad de días para simular)

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


#___________________________________________
# Valores para el Cluster:
#___________________________________________
"""
deltat_max = 300 s     # 5 min
deltat0 = 60 s
t_end = 30 * 86400 s   # 30 días


periods_save.phys_fields = 6 * 3600      # cada 6 h
periods_save.spect_energy_budg = 3600
"""







