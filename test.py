from fluidsim.solvers.ns2d.solver import Simul
from fluiddyn.util.mpi import rank

# Crear parámetros por defecto
params = Simul.create_default_params()

# Activar forcing coarse
params.forcing.enable = True
params.forcing.type = "in_script_coarse"

# Crear simulación
sim = Simul(params)

# Obtener objeto forcing
forcing_maker = sim.forcing.forcing_maker
if hasattr(forcing_maker, "oper_coarse"):

    oper_coarse = forcing_maker.oper_coarse

    if rank == 0:
        print("\nUsing oper_coarse")

else:

    oper_coarse = forcing_maker.oper

    if rank == 0:
        print("\nUsing oper")
# Mostrar información del objeto
if rank == 0:

    print("\n==============================")
    print("TYPE")
    print("==============================")
    print(type(forcing_maker))

    print("\n==============================")
    print("DIR")
    print("==============================")

    attrs = [a for a in dir(forcing_maker)
             if not a.startswith("__")]

    for attr in attrs:
        print(attr)

    print("\n==============================")
    print("__dict__.keys()")
    print("==============================")

    print(forcing_maker.__dict__.keys())
