#!/usr/bin/env python3
"""
Post‑procesamiento de la simulación FluidSim:
Estudio de la velocidad máxima a lo largo del tiempo.

Uso:
    python postprocess_velocity_max.py /ruta/al/directorio_de_salida

Si no se da ruta, se usa el directorio actual.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from fluidsim import load_sim_for_plot

# ============================================================
# 1. Cargar la simulación
# ============================================================
def load_simulation(path_run=None):
    """Carga el objeto Simul desde el directorio de resultados."""
    if path_run is None:
        path_run = os.getcwd()
    sim = load_sim_for_plot(path_run)
    print(f"Simulación cargada desde: {path_run}")
    print(f"Resolución: {sim.oper.nx} x {sim.oper.ny}")
    return sim

# ============================================================
# 2. Extraer velocidad máxima en función del tiempo
# ============================================================
def get_velocity_max_over_time(sim):
    """
    Recorre todos los tiempos guardados y calcula el máximo espacial
    de la magnitud de la velocidad |u| = sqrt(ux² + uy²).
    Devuelve arrays: times, ux_max, uy_max, umag_max.
    """
    # Actualizar lista de tiempos
    sim.output.phys_fields.set_of_phys_files.update_times()
    times = sim.output.phys_fields.set_of_phys_files.times
    print(f"Número de tiempos: {len(times)}")

    ux_max = np.zeros_like(times)
    uy_max = np.zeros_like(times)
    umag_max = np.zeros_like(times)

    for i, t in enumerate(times):
        # Cargar componentes
        res_ux = sim.output.phys_fields.get_field_to_plot(time=t, key='ux')
        res_uy = sim.output.phys_fields.get_field_to_plot(time=t, key='uy')
        # Desempaquetar (puede ser tupla)
        ux = res_ux[0] if isinstance(res_ux, tuple) else res_ux
        uy = res_uy[0] if isinstance(res_uy, tuple) else res_uy

        # Magnitud
        umag = np.sqrt(ux**2 + uy**2)

        # Máximos espaciales
        ux_max[i] = np.max(np.abs(ux))
        uy_max[i] = np.max(np.abs(uy))
        umag_max[i] = np.max(umag)

        if i % 100 == 0:
            print(f"Procesado tiempo {i+1}/{len(times)}: t = {t/86400:.2f} días")

    return times, ux_max, uy_max, umag_max

# ============================================================
# 3. Graficar
# ============================================================
def plot_velocity_max(times, ux_max, uy_max, umag_max, savefig=True):
    """Genera gráfica de la evolución de las velocidades máximas."""
    plt.figure(figsize=(12, 6))

    plt.plot(times/86400, ux_max, label=r'máx |$u$|', linewidth=2)
    plt.plot(times/86400, uy_max, label=r'máx |$v$|', linewidth=2)
    plt.plot(times/86400, umag_max, label=r'máx |$\mathbf{u}$|', linewidth=2, linestyle='--')

    plt.xlabel('Tiempo [días]')
    plt.ylabel('Velocidad máxima [m/s]')
    plt.title('Evolución de la velocidad máxima en el dominio')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig('velocity_max_evolution.png', dpi=150)
        print("Figura guardada como 'velocity_max_evolution.png'")
    plt.show()

# ============================================================
# 4. Función principal
# ============================================================
def main(path_run=None):
    sim = load_simulation(path_run)
    times, ux_max, uy_max, umag_max = get_velocity_max_over_time(sim)
    plot_velocity_max(times, ux_max, uy_max, umag_max)

    # Imprimir estadísticas
    print("\n--- Estadísticas ---")
    print(f"Velocidad máxima total alcanzada: {umag_max.max():.2f} m/s en t = {times[np.argmax(umag_max)]/86400:.2f} días")
    print(f"Velocidad |u| máxima: {ux_max.max():.2f} m/s")
    print(f"Velocidad |v| máxima: {uy_max.max():.2f} m/s")

if __name__ == "__main__":
    # Si se pasa un argumento, se toma como directorio de la simulación
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
