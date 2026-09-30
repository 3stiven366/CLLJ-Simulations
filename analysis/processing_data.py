import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin pantalla: necesario en el cluster (sin display)
import matplotlib.pyplot as plt

from Load_Physics_State import (
    load_full_simulation, estimate_memory_gb,
    load_spatial_means, load_jet_diagnostics, load_wave_catalog,
    load_spectra1D, load_spectra2D, load_spect_energy_budg,
    load_config, load_params_xml,
)

# ---------------------------------------------------------------
# Guard de MPI: si esto se lanza con mpirun -np N (p. ej. por la política
# del cluster), solo rank 0 hace el análisis; los demás ranks salen de
# inmediato. El análisis NO tiene comunicación colectiva, así que es seguro
# que salgan. Sin MPI (laptop/serial) cae a rank = 0 y corre normal.
# ---------------------------------------------------------------
try:
    from fluiddyn.util.mpi import rank
except ImportError:
    rank = 0

if rank != 0:
    sys.exit(0)

# ---------------------------------------------------------------
# Compatibilidad trapz / trapezoid (numpy <2 usa trapz, >=2 usa trapezoid)
# ---------------------------------------------------------------
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

# ---------------------------------------------------------------
# CONSTANTES FÍSICAS  (tomadas de CLLJ_simulation.py para ser consistentes)
# ---------------------------------------------------------------
PHI_REF   = 15.0
M_LAT     = 111e3                          # m por grado de latitud
M_LON     = 111e3 * np.cos(np.radians(PHI_REF))  # m por grado de longitud (~107.2 km)
LAT_MIN   = 5.0                            # °N en y = 0
Ly_deg    = 20.0                           # extensión meridional del dominio [°]
LAT_MAX   = LAT_MIN + Ly_deg               # 25 °N
TAPER_DEG = 3.0                            # ancho del taper en cada borde [°]
beta      = 2.29e-11                       # [s^-1 m^-1]


# ---------------------------------------------------------------
# Carpeta de salida en ~/ : Experimento1, Experimento2, ...
# ---------------------------------------------------------------
def make_experiment_dir(base=None):
    if base is None:
        base = os.path.expanduser("~")
    i = 1
    while True:
        path = os.path.join(base, f"Experimento{i}")
        if not os.path.exists(path):
            os.makedirs(path)
            return path
        i += 1


def shade_taper(ax):
    """Sombrea las zonas de taper (bordes) en un eje cuyo eje-y es latitud."""
    ax.axhspan(LAT_MIN, LAT_MIN + TAPER_DEG, color="gray", alpha=0.12)
    ax.axhspan(LAT_MAX - TAPER_DEG, LAT_MAX, color="gray", alpha=0.12)


# ===============================================================
# 1) CARGA DE DATOS
# ===============================================================
default_dir = "/home/Estiven/Sim_data/barotropic_cllj_W_N/NS2D_256x128_S4288710.669x2220000_2026-09-08_23-12-16"
directory = sys.argv[1] if len(sys.argv) > 1 else default_dir

estimate_memory_gb(directory)
x, y, times, u, v, rot = load_full_simulation(directory)

# ===============================================================
# 2) DESCOMPOSICIÓN DE REYNOLDS
# ===============================================================
u_xmean = u.mean(axis=2, keepdims=True)   # [u](t,y,1)
v_xmean = v.mean(axis=2, keepdims=True)   # [v](t,y,1)
u_prime = u - u_xmean                      # u'(t,y,x)
v_prime = v - v_xmean                      # v'(t,y,x)

u_profile = u.mean(axis=(0, 2))            # ū(y)  (ny,)


# ===============================================================
# 3) FUNCIONES DE DIAGNÓSTICO (basadas en los campos)
# ===============================================================
def eddy_momentum_flux(u_prime, v_prime):
    """Flujo de momento meridional [u'v'](t, y)  ->  (nt, ny)"""
    return (u_prime * v_prime).mean(axis=2)


def eddy_kinetic_energy(u_prime, v_prime, y):
    """
    EKE(t,y)   energía cinética de los torbellinos (media zonal)  -> (nt, ny)
    EKE_int(t) integrada en latitud                               -> (nt,)
    """
    eke = 0.5 * (u_prime**2 + v_prime**2).mean(axis=2)   # (nt, ny)
    eke_int = _trapz(eke, y, axis=1)                     # (nt,)
    return eke, eke_int


def barotropic_conversion(u_prime, v_prime, u_xmean, y):
    """
    CBT(t,y) = -[u'v'] * d[u]/dy   conversión barotrópica  -> (nt, ny)
    CBT_int(t) integrada en latitud                        -> (nt,)
    """
    uv = (u_prime * v_prime).mean(axis=2)             # (nt, ny)
    dudy = np.gradient(u_xmean[:, :, 0], y, axis=1)   # (nt, ny)
    cbt = -uv * dudy                                   # (nt, ny)
    cbt_int = _trapz(cbt, y, axis=1)                  # (nt,)
    return cbt, cbt_int


def rayleigh_kuo(beta, u_profile, y):
    """
    dq/dy = beta - d^2(ū)/dy^2   -> (ny,)
    Condición necesaria: dq/dy debe CAMBIAR DE SIGNO en el dominio.
    """
    d2u = np.gradient(np.gradient(u_profile, y), y)   # d^2 ū / dy^2  (ny,)
    return beta - d2u


def fjortoft(dqdy, u_profile):
    """
    Criterio de Fjortoft (refuerza a Rayleigh-Kuo).
    Existe alguna región donde  dq/dy * (ū - ū_s) < 0 ,
    con ū_s = ū en la latitud del punto de inflexión (dq/dy = 0).
    Devuelve: (u_s, region_bool, satisfecho_bool)
    """
    sign = np.sign(dqdy)
    crossings = np.where(np.diff(sign) != 0)[0]
    if crossings.size == 0:
        return np.nan, np.zeros_like(dqdy, dtype=bool), False
    i0 = crossings[0]
    denom = dqdy[i0] - dqdy[i0 + 1]
    frac = dqdy[i0] / denom if denom != 0 else 0.0
    u_s = u_profile[i0] + frac * (u_profile[i0 + 1] - u_profile[i0])
    region = dqdy * (u_profile - u_s) < 0
    return u_s, region, bool(region.any())


# ===============================================================
# 4) CÁLCULO DE DIAGNÓSTICOS (campos)
# ===============================================================
uv = eddy_momentum_flux(u_prime, v_prime)                          # (nt, ny)
eke, eke_int = eddy_kinetic_energy(u_prime, v_prime, y)            # (nt,ny), (nt,)
cbt, cbt_int = barotropic_conversion(u_prime, v_prime, u_xmean, y) # (nt,ny), (nt,)
dqdy = rayleigh_kuo(beta, u_profile, y)                            # (ny,)

# Perfiles promediados en el tiempo
uv_time   = uv.mean(axis=0)               # [u'v'](y)
cbt_time  = cbt.mean(axis=0)              # CBT(y)
eke_time  = eke.mean(axis=0)             # EKE(y)
dudy_time = np.gradient(u_profile, y)    # dū/dy (y)
eke_domain_mean = eke.mean(axis=1)       # EKE media de dominio(t) [m^2/s^2] (para validación cruzada)

cbt_total = _trapz(cbt_time, y)          # escalar

# --- Latitud y máscara del interior (excluye el taper) ---
lat = LAT_MIN + y / M_LAT                                  # °N
interior = (lat >= LAT_MIN + TAPER_DEG) & (lat <= LAT_MAX - TAPER_DEG)

# Criterios evaluados SOLO en el interior físico (el taper introduce
# curvatura artificial en los bordes que no es inestabilidad real)
dqdy_int = dqdy[interior]
lat_int  = lat[interior]
u_prof_int = u_profile[interior]

rk_cross_idx = np.where(np.diff(np.sign(dqdy_int)) != 0)[0]
rk_ok = rk_cross_idx.size > 0
rk_cross_lats = lat_int[rk_cross_idx]

u_s, fjortoft_region_int, fjortoft_ok = fjortoft(dqdy_int, u_prof_int)
# region de Fjortoft mapeada al arreglo completo (para graficar)
fjortoft_region = np.zeros_like(dqdy, dtype=bool)
fjortoft_region[np.where(interior)[0][fjortoft_region_int]] = True


# --- Criterios RESUELTOS EN EL TIEMPO (por snapshot) ---
# El promedio temporal de 90 días "lava" el perfil del jet (se relaja mucho
# durante la corrida), así que el criterio necesario se evalúa AQUÍ snapshot a
# snapshot sobre el estado básico instantáneo ū(y,t), sin promediar ni elegir
# ventanas arbitrarias.
SMOOTH_LAT = 5   # ventana (nº de puntos) para suavizar ū(y) antes de la 2ª derivada; 1 = sin suavizar

def _smooth_last_axis(arr, window):
    """Media móvil a lo largo del último eje (latitud). window<=1 -> sin cambio."""
    if window <= 1:
        return arr
    k = np.ones(window) / window
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), -1, arr)

ubar_yt = u.mean(axis=2)                             # ū(y,t)  (nt, ny)
ubar_sm = _smooth_last_axis(ubar_yt, SMOOTH_LAT)     # suavizado en latitud
d2u_yt = np.gradient(np.gradient(ubar_sm, y, axis=1), y, axis=1)  # ∂²ū/∂y² (nt, ny)
dqdy_yt = beta - d2u_yt                              # ∂q̄/∂y (y,t)  (nt, ny)

# ¿en cada snapshot cambia de signo dq/dy dentro del interior? -> booleano (nt,)
sign_int = np.sign(dqdy_yt[:, interior])
rk_sat_t = (np.diff(sign_int, axis=1) != 0).any(axis=1)   # (nt,) True = Rayleigh-Kuo satisfecho

# Fjortoft por snapshot (usa la función existente sobre el interior de cada t)
fj_sat_t = np.zeros(len(times), dtype=bool)
for it in range(len(times)):
    _, _, ok_fj = fjortoft(dqdy_yt[it, interior], ubar_yt[it, interior])
    fj_sat_t[it] = ok_fj

def _true_intervals(mask, t_days):
    """Devuelve lista de (t_ini, t_fin) en días de los tramos contiguos True."""
    intervals = []
    in_run = False
    for i, v in enumerate(mask):
        if v and not in_run:
            start = t_days[i]; in_run = True
        elif not v and in_run:
            intervals.append((start, t_days[i-1])); in_run = False
    if in_run:
        intervals.append((start, t_days[-1]))
    return intervals

rk_intervals = _true_intervals(rk_sat_t, times / 86400.0)
fj_intervals = _true_intervals(fj_sat_t, times / 86400.0)


# ===============================================================
# 5) FIGURAS BASADAS EN LOS CAMPOS  (ahora en latitud real)
# ===============================================================
outdir = make_experiment_dir()
print(f"Guardando figuras en: {outdir}")

times_d = times / 86400.0   # días
dlat = lat[1] - lat[0]

# --- Fig 1: perfil del jet ū(y) + EKE(y) ---
fig, ax1 = plt.subplots(figsize=(6, 5))
ax1.plot(u_profile, lat, color="navy", label=r"$\bar{u}$")
ax1.axvline(0, color="gray", lw=0.5)
shade_taper(ax1)
ax1.set_xlabel(r"$\bar{u}$ [m/s]", color="navy")
ax1.set_ylabel("latitud [°N]")
ax1.tick_params(axis="x", labelcolor="navy")
ax2 = ax1.twiny()
ax2.plot(eke_time, lat, color="crimson", label="EKE(y)")
ax2.set_xlabel(r"EKE [m$^2$/s$^2$]", color="crimson")
ax2.tick_params(axis="x", labelcolor="crimson")
ax1.set_title("Perfil del jet y EKE (zona gris = taper)")
fig.tight_layout()
fig.savefig(os.path.join(outdir, "01_perfil_jet_EKE.png"), dpi=150)
plt.close(fig)

# --- Fig 2: cizalladura dū/dy(y) ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(lat, dudy_time, height=dlat, color="teal")
ax.axvline(0, color="k", lw=0.5)
shade_taper(ax)
ax.set_xlabel(r"$\partial \bar{u}/\partial y$ [s$^{-1}$]")
ax.set_ylabel("latitud [°N]")
ax.set_title("Cizalladura del flujo medio")
fig.tight_layout()
fig.savefig(os.path.join(outdir, "02_cizalladura.png"), dpi=150)
plt.close(fig)

# --- Fig 3: flujo de momento [u'v'](y) ---
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh(lat, uv_time, height=dlat, color="darkorange")
ax.axvline(0, color="k", lw=0.5)
shade_taper(ax)
ax.set_xlabel(r"$\langle u'v' \rangle$ [m$^2$/s$^2$]")
ax.set_ylabel("latitud [°N]")
ax.set_title("Flujo de momento meridional de los torbellinos")
fig.tight_layout()
fig.savefig(os.path.join(outdir, "03_flujo_momento.png"), dpi=150)
plt.close(fig)

# --- Fig 4: conversión barotrópica CBT(y) ---
fig, ax = plt.subplots(figsize=(6, 5))
colors = ["crimson" if c > 0 else "steelblue" for c in cbt_time]
ax.barh(lat, cbt_time, height=dlat, color=colors)
ax.axvline(0, color="k", lw=0.5)
shade_taper(ax)
ax.set_xlabel(r"CBT [m$^2$/s$^3$]")
ax.set_ylabel("latitud [°N]")
ax.set_title("Conversión barotrópica\n(rojo: jet->ondas, azul: ondas->jet)")
fig.tight_layout()
fig.savefig(os.path.join(outdir, "04_CBT_perfil.png"), dpi=150)
plt.close(fig)

# --- Fig 5: Hovmoller de EKE(t, y) ---
fig, ax = plt.subplots(figsize=(8, 5))
pc = ax.pcolormesh(times_d, lat, eke.T, shading="auto", cmap="viridis")
fig.colorbar(pc, ax=ax, label=r"EKE [m$^2$/s$^2$]")
ax.set_xlabel("tiempo [días]")
ax.set_ylabel("latitud [°N]")
ax.set_title("Hovmoller de EKE")
fig.tight_layout()
fig.savefig(os.path.join(outdir, "05_hovmoller_EKE.png"), dpi=150)
plt.close(fig)

# --- Fig 6: presupuesto energético temporal (EKE_int y CBT_int) ---
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(times_d, eke_int, color="crimson", label=r"EKE$_{int}$(t)")
ax1.set_xlabel("tiempo [días]")
ax1.set_ylabel(r"EKE$_{int}$ [m$^3$/s$^2$]", color="crimson")
ax1.tick_params(axis="y", labelcolor="crimson")
ax2 = ax1.twinx()
ax2.plot(times_d, cbt_int, color="navy", label=r"CBT$_{int}$(t)")
ax2.axhline(0, color="gray", lw=0.5)
ax2.set_ylabel(r"CBT$_{int}$ [m$^3$/s$^3$]", color="navy")
ax2.tick_params(axis="y", labelcolor="navy")
ax1.set_title("Presupuesto energético temporal (campos)")
fig.tight_layout()
fig.savefig(os.path.join(outdir, "06_presupuesto_temporal.png"), dpi=150)
plt.close(fig)

# --- Fig 7: Rayleigh-Kuo + Fjortoft (interior, taper sombreado) ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(dqdy, lat, color="purple", label=r"$\partial \bar{q}/\partial y$")
ax.axvline(0, color="red", ls=":", label="cero")
shade_taper(ax)
if fjortoft_ok:
    ax.fill_betweenx(lat, dqdy.min(), dqdy.max(),
                     where=fjortoft_region, color="gold", alpha=0.25,
                     label="región Fjortoft")
for lat_c in rk_cross_lats:
    ax.axhline(lat_c, color="green", lw=0.8, ls="--", alpha=0.7)
ax.set_xlabel(r"$\partial \bar{q}/\partial y$ [m$^{-1}$s$^{-1}$]")
ax.set_ylabel("latitud [°N]")
ax.set_title("Rayleigh-Kuo y Fjortoft — PROMEDIO 90 días (referencia)\n(el jet se relaja; ver figs 14-15 para la evolución)")
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "07_rayleigh_kuo_fjortoft.png"), dpi=150)
plt.close(fig)

# --- Fig 14: mapa 2D de dq/dy(lat, t) — evolución del criterio, sin promediar ---
fig, ax = plt.subplots(figsize=(9, 5.5))
vmax = np.nanpercentile(np.abs(dqdy_yt), 99)
pc = ax.pcolormesh(times_d, lat, dqdy_yt.T, shading="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
# contorno del cero: donde dq/dy=0 -> punto de inflexión (condición necesaria)
ax.contour(times_d, lat, dqdy_yt.T, levels=[0], colors="k", linewidths=0.8)
ax.axhline(LAT_MIN + TAPER_DEG, color="gray", ls="--", lw=0.7)
ax.axhline(LAT_MAX - TAPER_DEG, color="gray", ls="--", lw=0.7)
ax.set_xlabel("tiempo [días]"); ax.set_ylabel("latitud [°N]")
ax.set_title(r"$\partial \bar{q}/\partial y\,(lat, t)$ — línea negra = cero (inflexión)")
fig.colorbar(pc, ax=ax, label=r"$\partial \bar{q}/\partial y$ [m$^{-1}$s$^{-1}$]")
fig.savefig(os.path.join(outdir, "14_rayleigh_kuo_tiempo.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# --- Fig 15: criterio satisfecho (sí/no) vs tiempo, junto a la EKE ---
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.fill_between(times_d, 0, 1, where=rk_sat_t, color="green", alpha=0.25,
                 transform=ax1.get_xaxis_transform(), label="Rayleigh-Kuo satisfecho")
ax1.fill_between(times_d, 0, 1, where=fj_sat_t, color="gold", alpha=0.35,
                 transform=ax1.get_xaxis_transform(), label="Fjortoft satisfecho")
ax1.plot(times_d, eke_int, color="crimson", lw=1.2, label=r"EKE$_{int}$(t)")
ax1.set_xlabel("tiempo [días]")
ax1.set_ylabel(r"EKE$_{int}$ [m$^3$/s$^2$]", color="crimson")
ax1.tick_params(axis="y", labelcolor="crimson")
ax1.set_title("Criterio de inestabilidad vs. crecimiento de las ondas")
ax1.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "15_criterio_vs_EKE.png"), dpi=150)
plt.close(fig)


# ===============================================================
# 6) FIGURAS DE LOS DIAGNÓSTICOS DE FLUIDSIM (punto 1)
#    Cada bloque va protegido: si falta un archivo, avisa y sigue.
# ===============================================================

# --- Fig 8: energía y enstrofía globales (spatial_means) ---
spatial_means = None
try:
    spatial_means = load_spatial_means(directory)
    sm_t = spatial_means["time"] / 86400.0
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4))
    axa.plot(sm_t, spatial_means["E"], "o-", color="darkred")
    axa.set_xlabel("tiempo [días]"); axa.set_ylabel("E [m$^2$/s$^2$]")
    axa.set_title("Energía cinética global")
    axb.plot(sm_t, spatial_means["epsK_tot"], "o-", color="teal", label="disipación epsK")
    axb.plot(sm_t, spatial_means["PK_tot"], "s-", color="darkorange", label="inyección PK")
    axb.axhline(0, color="gray", lw=0.5)
    axb.set_xlabel("tiempo [días]"); axb.set_ylabel("[m$^2$/s$^3$]")
    axb.set_title("Disipación vs. inyección"); axb.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "08_spatial_means.png"), dpi=150)
    plt.close(fig)
except Exception as e:
    print(f"[aviso] no se pudo graficar spatial_means: {e}")

# --- Fig 9: espectros 1D E(kx) y E(ky) — color por tiempo + colorbar (sin leyenda) ---
spec1d = None
try:
    spec1d = load_spectra1D(directory)
    t_d = spec1d["times"] / 86400.0
    nt_s = len(t_d)
    norm = plt.Normalize(t_d.min(), t_d.max())
    cmap = plt.cm.viridis
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i in range(nt_s):
        c = cmap(norm(t_d[i]))
        axa.loglog(spec1d["kx"][1:], spec1d["Ekx"][i, 1:], color=c, lw=0.7, alpha=0.7)
        axb.loglog(spec1d["ky"][1:], spec1d["Eky"][i, 1:], color=c, lw=0.7, alpha=0.7)
    axa.set_xlabel(r"$k_x$ [rad/m]"); axa.set_ylabel("E"); axa.set_title("Espectro E(kx)")
    axb.set_xlabel(r"$k_y$ [rad/m]"); axb.set_title("Espectro E(ky)")
    # piso en el eje y para no mostrar el "ruido" numérico de los modos vacíos
    for ax in (axa, axb):
        top = max(spec1d["Ekx"].max(), spec1d["Eky"].max())
        ax.set_ylim(top * 1e-8, top * 3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=[axa, axb], label="tiempo [días]")
    fig.savefig(os.path.join(outdir, "09_espectros_1D.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print(f"[aviso] no se pudo graficar spectra1D: {e}")

# --- Fig 10: espectro 2D E(kh) — color por tiempo + colorbar, con piso en y y ref k^-3 ---
spec2d = None
try:
    spec2d = load_spectra2D(directory)
    t_d = spec2d["times"] / 86400.0
    nt_s = len(t_d)
    norm = plt.Normalize(t_d.min(), t_d.max())
    cmap = plt.cm.plasma
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i in range(nt_s):
        ax.loglog(spec2d["kh"][1:], spec2d["E2D"][i, 1:],
                  color=cmap(norm(t_d[i])), lw=0.7, alpha=0.7)
    # referencia k^-3 (turbulencia 2D, cascada de enstrofía) — solo guía visual
    kk = spec2d["kh"][1:]
    ref = spec2d["E2D"][-1, 1] * (kk / kk[0])**(-3.0)
    ax.loglog(kk, ref, "k--", lw=1.2, label=r"$k^{-3}$ (ref.)")
    top = spec2d["E2D"].max()
    ax.set_ylim(top * 1e-8, top * 3)   # piso: oculta el ruido de modos casi vacíos
    ax.set_xlabel(r"$k_h$ [rad/m]"); ax.set_ylabel("E(kh)")
    ax.set_title("Espectro 2D de energía"); ax.legend(fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="tiempo [días]")
    fig.savefig(os.path.join(outdir, "10_espectro_2D.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print(f"[aviso] no se pudo graficar spectra2D: {e}")

# --- Fig 11: transferencia espectral como MAPA 2D (kh vs tiempo, color = transferencia) ---
seb = None
try:
    seb = load_spect_energy_budg(directory)
    t_d = seb["times"] / 86400.0
    kh = seb["kh"][1:]                       # ignora kh=0
    T = seb["transfer_E"][:, 1:]             # (nt, nkh-1)
    # escala de color simétrica en torno a 0 (transferencia cambia de signo)
    vmax = np.nanpercentile(np.abs(T), 99)   # percentil 99 para que un outlier no aplaste el color
    fig, ax = plt.subplots(figsize=(9, 5.5))
    pc = ax.pcolormesh(kh, t_d, T, shading="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k_h$ [rad/m]"); ax.set_ylabel("tiempo [días]")
    ax.set_title("Presupuesto espectral de energía\n(rojo: la escala gana E, azul: pierde)")
    fig.colorbar(pc, ax=ax, label="transferencia de E")
    fig.savefig(os.path.join(outdir, "11_transferencia_espectral.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print(f"[aviso] no se pudo graficar spect_energy_budg: {e}")

# --- Fig 12 y 13: diagnósticos del jet (KE_zonal vs KE_eddy) + validación cruzada ---
jet_diag = None
try:
    jet_diag = load_jet_diagnostics(directory)
    if jet_diag["t_days"].size == 0:
        print("[aviso] jet_diagnostics vacío: se omiten Fig 12 y 13.")
    else:
        jt = jet_diag["t_days"]
        # Fig 12: intercambio jet <-> eddies
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(jt, jet_diag["KE_zonal"], color="navy", label="KE_zonal (jet)")
        ax1.set_xlabel("tiempo [días]"); ax1.set_ylabel("KE_zonal [m$^2$/s$^2$]", color="navy")
        ax1.tick_params(axis="y", labelcolor="navy")
        ax2 = ax1.twinx()
        ax2.plot(jt, jet_diag["KE_eddy"], color="crimson", label="KE_eddy (ondas)")
        ax2.set_ylabel("KE_eddy [m$^2$/s$^2$]", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson")
        ax1.set_title("Intercambio de energía jet <-> eddies")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "12_KE_zonal_vs_eddy.png"), dpi=150)
        plt.close(fig)

        # Fig 13: validación cruzada KE_eddy (simulación) vs EKE media de dominio (mi cálculo)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(jt, jet_diag["KE_eddy"], "o-", color="crimson", label="KE_eddy (jet_diagnostics)")
        ax.plot(times_d, eke_domain_mean, "s--", color="navy", label="EKE media de dominio (campos)")
        ax.set_xlabel("tiempo [días]"); ax.set_ylabel("[m$^2$/s$^2$]")
        ax.set_title("Validación cruzada: energía de los eddies")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "13_validacion_cruzada_EKE.png"), dpi=150)
        plt.close(fig)
except Exception as e:
    print(f"[aviso] no se pudo graficar jet_diagnostics: {e}")

# --- catálogo de ondas (para comparar escalas inyectadas con los espectros) ---
wave_cat = None
try:
    wave_cat = load_wave_catalog(directory)
except Exception as e:
    print(f"[aviso] no se pudo leer wave_catalog: {e}")


# ===============================================================
# 7) EVALUACIÓN DE CRITERIOS (impreso + guardado en resumen.txt)
# ===============================================================
L = []
L.append("=" * 60)
L.append("RESUMEN DE DIAGNÓSTICOS - INESTABILIDAD BAROTRÓPICA")
L.append("=" * 60)
L.append(f"Directorio de datos : {directory}")
L.append(f"Snapshots (nt)      : {len(times)}")
L.append(f"Malla (ny x nx)     : {len(y)} x {len(x)}")
L.append(f"Dominio             : {LAT_MIN:.0f}-{LAT_MAX:.0f} °N  (taper de {TAPER_DEG:.0f}° en cada borde)")
L.append(f"Interior analizado  : {LAT_MIN+TAPER_DEG:.0f}-{LAT_MAX-TAPER_DEG:.0f} °N")
L.append("")

# --- Configuración del experimento (config propia + params de FluidSim) ---
cfg = load_config(directory)
params_xml = load_params_xml(directory)
if cfg is not None or params_xml is not None:
    L.append("--- Configuracion del experimento ---")
    if cfg is not None:
        L.append(f"  Forzamiento de ondas : {cfg.get('ACTIVE_WAVES')}")
        L.append(f"  Nudging del jet      : {cfg.get('ACTIVE_JET_NUDGING')}")
        jp = cfg.get("JET_PARAMS")
        if jp:
            centros = ", ".join(f"{g[0]:.2f}" for g in jp)
            L.append(f"  Jet ({len(jp)} gaussianas)  : centros {centros} °N")
        L.append(f"  Ruido sintetico      : {cfg.get('NOISE_SIGMA')} m/s")
        if cfg.get("TAU_RELAX") is not None:
            L.append(f"  TAU_RELAX            : {cfg['TAU_RELAX']/86400:.1f} dias")
    if params_xml is not None:
        L.append(f"  (FluidSim) beta      : {params_xml.get('beta')}")
        L.append(f"  (FluidSim) nu_2      : {params_xml.get('nu_2'):.3e}")
        L.append(f"  (FluidSim) nu_4      : {params_xml.get('nu_4'):.3e}")
        L.append(f"  (FluidSim) malla     : {params_xml.get('nx')} x {params_xml.get('ny')}")
        L.append(f"  (FluidSim) forcing   : enable={params_xml.get('forcing_enable')}, "
                 f"type={params_xml.get('forcing_type')}, key={params_xml.get('key_forced')}")
        L.append(f"  (FluidSim) FFT       : {params_xml.get('type_fft')}")
    L.append("")

L.append("--- Rayleigh-Kuo (condición necesaria, interior, RESUELTO EN EL TIEMPO) ---")
frac_rk = 100.0 * rk_sat_t.mean()
L.append(f"  Satisfecho en {rk_sat_t.sum()}/{len(times)} snapshots ({frac_rk:.1f}% del tiempo)")
if rk_intervals:
    tramos = ", ".join(f"{a:.1f}-{b:.1f} d" for a, b in rk_intervals[:10])
    extra = "" if len(rk_intervals) <= 10 else f" (+{len(rk_intervals)-10} tramos)"
    L.append(f"  Intervalos: {tramos}{extra}")
else:
    L.append("  Nunca se satisface en el interior.")
L.append("")
L.append("--- Fjortoft (refuerzo, interior, RESUELTO EN EL TIEMPO) ---")
frac_fj = 100.0 * fj_sat_t.mean()
L.append(f"  Satisfecho en {fj_sat_t.sum()}/{len(times)} snapshots ({frac_fj:.1f}% del tiempo)")
if fj_intervals:
    tramos = ", ".join(f"{a:.1f}-{b:.1f} d" for a, b in fj_intervals[:10])
    extra = "" if len(fj_intervals) <= 10 else f" (+{len(fj_intervals)-10} tramos)"
    L.append(f"  Intervalos: {tramos}{extra}")
else:
    L.append("  Nunca se satisface en el interior.")
L.append("")
L.append("  (Nota: el promedio de 90 días 'lava' el jet; por eso el criterio")
L.append("   se evalua por snapshot. Ver figs 14-15.)")
L.append("")
L.append("--- Conversion barotropica (campos) ---")
L.append(f"  CBT_total (temporal + integral en y) = {cbt_total:.3e} m^3/s^3")
if cbt_total > 0:
    L.append("  CBT_total > 0 -> energia neta del JET hacia las ONDAS (inestabilidad activa)")
else:
    L.append("  CBT_total < 0 -> las ondas refuerzan el jet")
j_max = np.argmax(eke_time)
L.append(f"  Maximo de EKE(y) en {lat[j_max]:.2f} N")
L.append("")

if spatial_means is not None:
    E = spatial_means["E"]
    L.append("--- Energia global (spatial_means) ---")
    tendencia = "CRECE" if E[-1] > E[0] else "DECAE"
    L.append(f"  E: {E[0]:.4f} -> {E[-1]:.4f}  ({tendencia}, {100*(E[-1]-E[0])/E[0]:+.1f}%)")
    L.append(f"  Inyeccion neta media PK_tot = {np.mean(spatial_means['PK_tot']):.3e}")
    L.append("")

if spec1d is not None:
    Ekx_last = spec1d["Ekx"][-1, 1:]
    kx_pos = spec1d["kx"][1:]
    kpi = np.argmax(Ekx_last)
    kx_peak = kx_pos[kpi]
    lam_peak_km = 2 * np.pi / kx_peak / 1e3 if kx_peak > 0 else np.nan
    L.append("--- Espectro E(kx) (ultimo tiempo) ---")
    L.append(f"  Pico en kx = {kx_peak:.3e} rad/m  (lambda ~ {lam_peak_km:.0f} km)")
    if wave_cat is not None and wave_cat.get("wavelength_km", np.array([])).size > 0:
        inj = ", ".join(f"{w:.0f}" for w in wave_cat["wavelength_km"])
        L.append(f"  Longitudes de onda inyectadas (catalogo): {inj} km")
    L.append("")

if jet_diag is not None and jet_diag["t_days"].size > 0:
    kz, ke = jet_diag["KE_zonal"], jet_diag["KE_eddy"]
    L.append("--- Energia jet vs eddies (jet_diagnostics) ---")
    L.append(f"  KE_zonal: {kz[0]:.4f} -> {kz[-1]:.4f}")
    L.append(f"  KE_eddy : {ke[0]:.4f} -> {ke[-1]:.4f}")
    if kz[-1] < kz[0] and ke[-1] > ke[0]:
        L.append("  -> jet PIERDE energia y eddies GANAN: firma de inestabilidad barotropica")
    L.append("")

L.append("=" * 60)

resumen = "\n".join(L)
print(resumen)
with open(os.path.join(outdir, "resumen.txt"), "w") as f:
    f.write(resumen + "\n")

print(f"\nListo. Figuras y resumen guardados en: {outdir}")