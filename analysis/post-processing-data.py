"""
Post-procesamiento de la simulacion barotropica CLLJ <-> ondas.

Carga la simulacion UNA vez, calcula:
  1. Criterio de Rayleigh-Kuo  (dq/dy)
  2. CBT y EKE  (perfiles + serie temporal)
  3. Diagrama Hovmoller de v'
y guarda todas las figuras en una carpeta, lista para bajar con scp.

Uso:
    python postproc_barotropico.py /ruta/a/tu/run

Corre en un nodo del cluster sin pantalla (backend Agg -> solo PNG).
Baja despues solo las figuras:
    scp usuario@cluster:/ruta/a/tu/run/figuras/*.png .
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")          # CRITICO: sin ventanas graficas en el nodo
import matplotlib.pyplot as plt
import numpy as np
import fluidsim as fls

# ------------------------------------------------------------------
# 0. Configuracion
# ------------------------------------------------------------------
LAT0     = 15.0        # centro del dominio en latitud real [N]
M        = 111e3       # grados -> metros

# Ruta de la simulacion: argumento de linea de comando, o editar aca
if len(sys.argv) > 1:
    PATH_RUN = sys.argv[1]
else:
    PATH_RUN = "~/Sim_data/barotropic_cllj_cluster/NS2D_1024x512_S13320000x2220000_2026-07-11_22-10-03"   # <-- editar si no se pasa por CLI

# ------------------------------------------------------------------
# 1. Cargar la simulacion (una sola vez)
# ------------------------------------------------------------------
print(f"[load] cargando simulacion desde:\n       {PATH_RUN}")
sim = fls.load_sim_for_plot(PATH_RUN)

# Carpeta de salida para las figuras, dentro del propio run
OUTDIR = os.path.join(sim.output.path_run, "figuras")
os.makedirs(OUTDIR, exist_ok=True)
print(f"[out ] figuras -> {OUTDIR}")

# Tiempos guardados
sim.output.phys_fields.set_of_phys_files.update_times()
times = np.array(sim.output.phys_fields.set_of_phys_files.times)
print(f"[time] {len(times)} campos guardados: "
      f"t = {times.min():.0f} .. {times.max():.0f} s "
      f"({times.min()/86400:.2f} .. {times.max()/86400:.2f} dias)")

t_last = float(times[-1])   # ultimo campo disponible (robusto a runs de cualquier duracion)

# OJO: tras load_sim_for_plot, sim.oper.y puede quedar en la grilla trivial
# (nx=ny=4) porque la carga no reconstruye el operador a resolucion completa.
# Los campos leidos del disco SI vienen a resolucion real, asi que la
# coordenada meridional se construye desde Ly y ny reales de los parametros.
Ly_real = sim.params.oper.Ly
ny_real = sim.params.oper.ny
dy      = Ly_real / ny_real
y_phys  = np.arange(ny_real) * dy          # 0 .. Ly (metros)
y_deg   = (y_phys - Ly_real / 2) / M + LAT0   # latitud real [N]
beta    = sim.params.beta

print(f"[grid] ny={ny_real}  Ly={Ly_real:.0f} m  dy={dy:.1f} m  "
      f"lat {y_deg.min():.1f}-{y_deg.max():.1f} N")


# ------------------------------------------------------------------
# Helper: cargar un campo desempaquetando la tupla si hace falta
# ------------------------------------------------------------------
def load_field(key, time):
    result = sim.output.phys_fields.get_field_to_plot(time=time, key=key)
    return result[0] if isinstance(result, tuple) else result


# ==================================================================
# 1. CRITERIO DE RAYLEIGH-KUO
# ==================================================================
print("\n[1/3] Rayleigh-Kuo ...")

ux     = load_field("ux", t_last)
u_mean = ux.mean(axis=1)                 # u_bar(y): promedio zonal

dudy    = np.gradient(u_mean, dy)
d2udy2  = np.gradient(dudy, dy)
PV_grad = beta - d2udy2                   # dq/dy

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(PV_grad, y_deg, "b-", lw=2, label=r"$\partial \bar{q}/\partial y$")
ax.axvline(0, color="k", ls="--", alpha=0.7, label="Criterio R-K")
ax.set_xlabel(r"$\partial \bar{q}/\partial y$  [m$^{-1}$s$^{-1}$]")
ax.set_ylabel("Latitud [$^\\circ$N]")
ax.set_title(f"Criterio de Rayleigh-Kuo  (t = {t_last/86400:.1f} d)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "rayleigh_kuo.png"), dpi=150)
plt.close(fig)

# Diagnostico del cambio de signo
neg = np.where(PV_grad < 0)[0]
print("     " + "=" * 55)
if len(neg) > 0:
    cross = np.where(np.diff(np.sign(PV_grad)) != 0)[0]
    print(f"     dq/dy < 0 en {len(neg)}/{len(PV_grad)} puntos "
          f"({y_deg[neg].min():.1f} - {y_deg[neg].max():.1f} N)")
    if len(cross) > 0:
        lats_cross = ", ".join(f"{y_deg[i]:.1f}N" for i in cross)
        print(f"     cruce(s) de signo en: {lats_cross}")
    print("     -> CONDICION NECESARIA de inestabilidad CUMPLIDA")
else:
    print("     dq/dy > 0 en todo el dominio")
    print("     -> flujo ESTABLE segun criterio necesario de R-K")
print(f"     dq/dy: min {PV_grad.min():.3e}  max {PV_grad.max():.3e}")


# ==================================================================
# 2. CBT y EKE  (perfiles en t_last + serie temporal completa)
# ==================================================================
print("\n[2/3] CBT y EKE ...")

# --- perfiles en el ultimo tiempo ---
ux = load_field("ux", t_last)
uy = load_field("uy", t_last)

u_mean  = ux.mean(axis=1)
u_prime = ux - u_mean[:, None]
v_prime = uy                              # v_bar ~ 0

uv       = (u_prime * v_prime).mean(axis=1)      # <u'v'>(y)
dudy     = np.gradient(u_mean, dy)
CBT_prof = -uv * dudy                            # CBT(y)
EKE_prof = 0.5 * (u_prime**2 + v_prime**2).mean(axis=1)   # EKE(y)  <- axis=1

# Panel doble: EKE(y) y CBT(y)
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
ax[0].plot(EKE_prof, y_deg, "g-", lw=2)
ax[0].set_xlabel(r"EKE [m$^2$/s$^2$]")
ax[0].set_ylabel("Latitud [$^\\circ$N]")
ax[0].set_title("Energia cinetica de perturbacion")
ax[0].grid(alpha=0.3)

ax[1].plot(CBT_prof, y_deg, "r-", lw=2)
ax[1].axvline(0, color="k", ls="--", alpha=0.7)
ax[1].set_xlabel(r"CBT [m$^2$/s$^3$]")
ax[1].set_title("Conversion barotropica  (>0: jet -> onda)")
ax[1].grid(alpha=0.3)
fig.suptitle(f"t = {t_last/86400:.1f} dias")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "cbt_eke_perfil.png"), dpi=150)
plt.close(fig)

# --- serie temporal: recorre todos los campos ---
EKE_t = np.empty(len(times))
CBT_t = np.empty(len(times))
for i, t in enumerate(times):
    ux = load_field("ux", float(t))
    uy = load_field("uy", float(t))
    um = ux.mean(axis=1)
    up = ux - um[:, None]
    vp = uy
    EKE_t[i] = 0.5 * (up**2 + vp**2).mean()                  # escalar (dominio)
    CBT_t[i] = np.trapz(-(up * vp).mean(axis=1) * np.gradient(um, dy), y_deg * M)

fig, ax1 = plt.subplots(figsize=(10, 6))
c1 = "tab:green"
ax1.plot(times / 86400, EKE_t, "-", color=c1, lw=2)
ax1.set_xlabel("Tiempo [dias]")
ax1.set_ylabel("EKE [m$^2$/s$^2$]", color=c1)
ax1.tick_params(axis="y", labelcolor=c1)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
c2 = "tab:red"
ax2.plot(times / 86400, CBT_t, "--", color=c2, lw=2)
ax2.set_ylabel(r"$\int$ CBT dy  [m$^3$/s$^3$]", color=c2)
ax2.tick_params(axis="y", labelcolor=c2)
ax2.axhline(0, color="k", lw=0.8, alpha=0.5)

fig.suptitle("Presupuesto energetico temporal")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "cbt_eke_tiempo.png"), dpi=150)
plt.close(fig)

print(f"     EKE(t_last) = {EKE_t[-1]:.4f} m2/s2")
print(f"     CBT(y) max  = {CBT_prof.max():.3e} m2/s3")
print(f"     int CBT dy  = {CBT_t[-1]:.3e} m3/s3  "
      f"({'jet->onda' if CBT_t[-1] > 0 else 'onda->jet'})")


# ==================================================================
# 3. DIAGRAMA HOVMOLLER de v'
# ==================================================================
print("\n[3/3] Hovmoller de v' ...")

# H[i, j] = v'(y_j) promediado zonalmente, en el tiempo t_i
H = np.empty((len(times), len(y_deg)))
for i, t in enumerate(times):
    uy = load_field("uy", float(t))
    H[i, :] = uy.mean(axis=1)              # <v'>(y) en cada tiempo

Y, T = np.meshgrid(y_deg, times / 86400)

fig, ax = plt.subplots(figsize=(11, 6))
vmax = np.abs(H).max()
pcm = ax.contourf(Y, T, H, levels=50, cmap="RdBu_r",
                  vmin=-vmax, vmax=vmax)
fig.colorbar(pcm, ax=ax, label=r"$\langle v' \rangle$ [m/s]")
ax.set_xlabel("Latitud [$^\\circ$N]")
ax.set_ylabel("Tiempo [dias]")
ax.set_title("Diagrama Hovmoller de v'")
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "hovmoller.png"), dpi=150)
plt.close(fig)

print(f"     H shape = {H.shape}  (tiempos x latitudes)")


# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("LISTO. Figuras guardadas en:")
print(f"  {OUTDIR}")
print("\nPara bajarlas a tu PC:")
print(f"  scp usuario@cluster:{OUTDIR}/*.png .")
print("=" * 60)
