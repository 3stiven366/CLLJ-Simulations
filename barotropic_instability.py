"""
=============================================================================
INTERCAMBIO DE ENERGÍA: JET ZONAL BAROTROPICO DEL ESTE vs ONDAS TROPICALES
=============================================================================
Estudia la interacción energética entre:
  - Flujo medio zonal (jet del este, u_bar < 0)
  - Perturbaciones de onda tropical del este (ondas easterly, ~2.5-6 días)

Marco teórico: Ecuación de energía barotropica (Andrews & McIntyre, 1976)
  dE'/dt = -<u'v'> * d(u_bar)/dy   (conversión barotropica)
  + términos de flujo y disipación

Variables:
  u : viento zonal (m/s)
  v : viento meridional (m/s)
  Dominio: tropical (±30°), p=850 hPa (nivel barotropico representativo)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import butter, filtfilt, welch
from scipy.ndimage import uniform_filter1d

# ──────────────────────────────────────────────────────────
# 1.  PARÁMETROS GENERALES
# ──────────────────────────────────────────────────────────
np.random.seed(42)

Ny, Nx, Nt = 60, 120, 180          # lat, lon, días
dy = 111_000                         # m por grado
dt = 86_400                          # s por día

lats = np.linspace(-30, 30, Ny)     # °N
lons = np.linspace(0, 360, Nx)
time = np.arange(Nt)                 # días

# ──────────────────────────────────────────────────────────
# 2.  CONSTRUCCIÓN DEL CAMPO SINTÉTICO REALISTA
# ──────────────────────────────────────────────────────────

def build_easterly_jet(lats):
    """Jet del este subtropical: máximo ~15°N, ~-8 m/s."""
    u_bar = np.zeros(len(lats))
    for lat, amp, sigma in [(-10, -6, 8), (10, -8, 7), (0, -3, 15)]:
        u_bar += amp * np.exp(-((lats - lat)**2) / (2 * sigma**2))
    return u_bar

u_bar_1d = build_easterly_jet(lats)   # (Ny,)
U_MEAN = np.tile(u_bar_1d, (Nx, 1)).T  # (Ny, Nx)


def easterly_wave(lats, lons, time, k=6, period=5, amp=3.0, lat0=10, sigma_y=8):
    """Onda tropical del este: k6, ~5 días, propagación hacia el oeste."""
    LO, LA, TI = np.meshgrid(lons, lats, time)
    phi = k * np.deg2rad(LO) + 2*np.pi * TI / period   # fase → oeste
    envelope = np.exp(-((LA - lat0)**2) / (2*sigma_y**2))
    u_prime = amp * np.sin(phi) * envelope
    v_prime = amp * 0.6 * np.cos(phi) * envelope
    return u_prime, v_prime


def add_noise(field, scale=0.5):
    return field + np.random.normal(0, scale, field.shape)


u_prime, v_prime = easterly_wave(lats, lons, time)
u_prime2, v_prime2 = easterly_wave(lats, lons, time, k=4, period=3.5,
                                    amp=1.5, lat0=-8, sigma_y=6)

u_prime = add_noise(u_prime + u_prime2)
v_prime = add_noise(v_prime + v_prime2)

# Campo total
U = U_MEAN[:, :, np.newaxis] + u_prime
V = v_prime

# ──────────────────────────────────────────────────────────
# 3.  FUNCIONES DE ANÁLISIS
# ──────────────────────────────────────────────────────────

def zonal_mean_and_eddy(field):
    """Descomposición [·] + (·)'  en la dirección zonal."""
    bar = np.mean(field, axis=1, keepdims=True)   # (Ny,1,Nt)
    prime = field - bar
    return bar[:, 0, :], prime    # bar:(Ny,Nt), prime:(Ny,Nx,Nt)


def barotropic_conversion(u_mean, u_p, v_p, lats):
    """
    C_BT = -<u'v'> * d(u_bar)/dy
    Conversión barotropica (positivo = ganancia de energía perturbación).
    """
    # Flujo de momento meridional zonal-mean temporal-mean
    uv_eddy = np.mean(u_p * v_p, axis=1)          # (Ny, Nt)
    uv_clim = np.mean(uv_eddy, axis=-1)            # (Ny,)
    
    # Gradiente meridional del viento medio temporal
    u_clim = np.mean(u_mean, axis=-1)              # (Ny,)
    du_dy = np.gradient(u_clim, np.deg2rad(lats) * 6.371e6)  # (Ny,)
    
    C_bt = -uv_clim * du_dy                        # (Ny,)
    return C_bt, uv_clim, u_clim, du_dy


def eddy_kinetic_energy(u_p, v_p):
    """EKE = 0.5*(u'^2 + v'^2) promedio zonal."""
    eke = 0.5 * (u_p**2 + v_p**2)
    return np.mean(eke, axis=1)   # (Ny, Nt)


def bandpass_filter(data, lowcut, highcut, fs=1.0, order=4):
    """Filtro paso-banda en la dimensión temporal."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)


def energy_budget_time_series(u_p, v_p, u_mean, lats):
    """
    Serie de tiempo del presupuesto de energía integrado meridionalmente.
    dEKE/dt ≈ C_BT + residuo
    """
    EKE = eddy_kinetic_energy(u_p, v_p)       # (Ny, Nt)
    EKE_zint = np.trapezoid(EKE, lats, axis=0)    # (Nt,)
    
    # C_BT en cada tiempo (variación temporal del estado medio suavizado)
    u_smooth = uniform_filter1d(u_mean, size=10, axis=-1)
    C_bt_ts = np.zeros(Nt)
    for t in range(Nt):
        uv_t = np.mean(u_p[:, :, t] * v_p[:, :, t], axis=1)
        u_t  = u_smooth[:, t]
        du_dy_t = np.gradient(u_t, np.deg2rad(lats) * 6.371e6)
        C_bt_t = -uv_t * du_dy_t
        C_bt_ts[t] = np.trapezoid(C_bt_t, lats)
    
    dEKE_dt = np.gradient(EKE_zint, dt)
    return EKE_zint, C_bt_ts, dEKE_dt


def power_spectrum(field_1d, fs=1.0):
    """Espectro de potencia Welch."""
    freqs, psd = welch(field_1d, fs=fs, nperseg=64)
    periods = 1.0 / (freqs + 1e-10)
    return periods, psd

# ──────────────────────────────────────────────────────────
# 4.  EJECUCIÓN DEL ANÁLISIS
# ──────────────────────────────────────────────────────────
print("=" * 60)
print("  ANÁLISIS DE INTERCAMBIO ENERGÉTICO")
print("  Jet Barotropico del Este ↔ Ondas Tropicales")
print("=" * 60)

u_mean, u_p = zonal_mean_and_eddy(U)
_,      v_p = zonal_mean_and_eddy(V)

C_bt, uv_mean, u_clim, du_dy = barotropic_conversion(u_mean, u_p, v_p, lats)
EKE_map = np.mean(eddy_kinetic_energy(u_p, v_p), axis=-1)
EKE_ts, C_bt_ts, dEKE_dt = energy_budget_time_series(u_p, v_p, u_mean, lats)

# Banda de onda tropical (2.5–6 días)
u_p_filt = bandpass_filter(u_p, 1/6, 1/2.5, fs=1.0)
v_p_filt = bandpass_filter(v_p, 1/6, 1/2.5, fs=1.0)
EKE_wave = np.mean(np.mean(eddy_kinetic_energy(u_p_filt, v_p_filt), axis=-1))

# Diagnóstico global
total_C_bt = np.trapezoid(C_bt, lats)
net_EKE = np.mean(EKE_ts)

print(f"\n  Jet del este (u_bar mín): {u_clim.min():.2f} m/s  @ {lats[np.argmin(u_clim)]:.1f}°N")
print(f"  EKE media:  {net_EKE/1e3:.1f} × 10³ m²/s²")
print(f"  EKE banda de ondas (2.5-6d): {EKE_wave:.2f} m²/s²")
print(f"  Conversión barotropica ∫C_BT dy: {total_C_bt:.4f} m³/s³")
print(f"  Signo C_BT: {'Flujo JET→ONDA (inestabilidad barotropica)' if total_C_bt > 0 else 'Flujo ONDA→JET'}")

# ──────────────────────────────────────────────────────────
# 5.  FIGURAS
# ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0e1117')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

DARK = '#0e1117'
PANEL = '#1c2333'
TEXT  = '#e8edf5'
ACC1  = '#4fc3f7'
ACC2  = '#ff7043'
ACC3  = '#66bb6a'

def style_ax(ax, title=''):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor('#3a4a6b')
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if title:
        ax.set_title(title, color=TEXT, fontsize=9, fontweight='bold', pad=6)

# ─── Panel 1: Perfil del jet + EKE
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'Perfil del Jet y EKE')
ax1.plot(u_clim, lats, color=ACC1, lw=2, label='ū (m/s)')
ax1.axvline(0, color='white', lw=0.5, ls='--', alpha=0.4)
ax1.set_xlabel('m/s'); ax1.set_ylabel('Latitud (°N)')
ax1r = ax1.twiny()
ax1r.plot(EKE_map, lats, color=ACC2, lw=2, ls='--', label='EKE')
ax1r.tick_params(colors=TEXT, labelsize=7)
ax1r.set_xlabel('EKE (m²/s²)', color=ACC2)
ax1.legend(loc='upper left', fontsize=7, facecolor=PANEL, labelcolor=TEXT)

# ─── Panel 2: Gradiente du/dy
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'd(ū)/dy — Región inestable')
norm = TwoSlopeNorm(vcenter=0)
ax2.barh(lats, du_dy * 1e6, color=[ACC2 if v > 0 else ACC1 for v in du_dy], height=0.9)
ax2.axvline(0, color='white', lw=1)
ax2.set_xlabel('d(ū)/dy × 10⁻⁶ s⁻¹')
ax2.set_ylabel('Latitud (°N)')

# ─── Panel 3: Conversión barotropica C_BT
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3, 'Conversión Barotropica C_BT')
colors_cbt = [ACC2 if v > 0 else ACC1 for v in C_bt]
ax3.barh(lats, C_bt, color=colors_cbt, height=0.9)
ax3.axvline(0, color='white', lw=1)
ax3.set_xlabel('C_BT (m²/s³ por grado)')
ax3.set_ylabel('Latitud (°N)')
ax3.text(0.97, 0.03, f'∫={total_C_bt:.3f}', transform=ax3.transAxes,
         color=ACC2 if total_C_bt > 0 else ACC1, ha='right', fontsize=8)

# ─── Panel 4: Mapa de EKE promedio temporal
ax4 = fig.add_subplot(gs[1, :2])
style_ax(ax4, 'EKE Media (banda 2.5–6 días) — lat-tiempo')
EKE_filt_ts = eddy_kinetic_energy(u_p_filt, v_p_filt)  # (Ny, Nt)
im4 = ax4.pcolormesh(time, lats, EKE_filt_ts, cmap='inferno', shading='auto')
plt.colorbar(im4, ax=ax4, label='EKE (m²/s²)', pad=0.01).ax.yaxis.label.set_color(TEXT)
ax4.set_xlabel('Tiempo (días)'); ax4.set_ylabel('Latitud (°N)')
ax4.contour(time, lats, np.tile(u_clim, (Nt,1)).T,
            levels=[-8,-6,-4,-2], colors='white', linewidths=0.6, alpha=0.5)

# ─── Panel 5: u'v' flujo de momento
ax5 = fig.add_subplot(gs[1, 2])
style_ax(ax5, "Flujo momento <u'v'>")
uv_lat = np.mean(np.mean(u_p * v_p, axis=1), axis=-1)
ax5.barh(lats, uv_lat, color=[ACC3 if v < 0 else ACC2 for v in uv_lat], height=0.9)
ax5.axvline(0, color='white', lw=1)
ax5.set_xlabel("<u'v'> (m²/s²)"); ax5.set_ylabel('Latitud (°N)')

# ─── Panel 6: Serie de tiempo EKE y C_BT
ax6 = fig.add_subplot(gs[2, :2])
style_ax(ax6, 'Presupuesto Energético Temporal (integrado en latitud)')
ax6.plot(time, EKE_ts / 1e3, color=ACC1, lw=1.5, label='EKE × 10³')
ax6r = ax6.twinx()
ax6r.plot(time, C_bt_ts, color=ACC2, lw=1.2, ls='--', label='C_BT (conversión)')
ax6r.axhline(0, color='white', lw=0.5, ls=':', alpha=0.4)
ax6r.tick_params(colors=TEXT, labelsize=7)
ax6r.set_ylabel('C_BT (m³/s³)', color=ACC2)
ax6.set_xlabel('Tiempo (días)')
ax6.set_ylabel('EKE × 10³ (m²/s²)', color=ACC1)
lines1, lab1 = ax6.get_legend_handles_labels()
lines2, lab2 = ax6r.get_legend_handles_labels()
ax6.legend(lines1+lines2, lab1+lab2, loc='upper right', fontsize=7,
           facecolor=PANEL, labelcolor=TEXT)
ax6r.spines['right'].set_edgecolor('#3a4a6b')
ax6r.yaxis.label.set_color(TEXT)

# ─── Panel 7: Espectro de potencia
ax7 = fig.add_subplot(gs[2, 2])
style_ax(ax7, 'Espectro de Potencia EKE')
i_jet = np.argmin(u_clim)
eke_1d = np.mean(eddy_kinetic_energy(u_p, v_p)[i_jet-3:i_jet+3, :], axis=0)
periods, psd = power_spectrum(eke_1d, fs=1.0)
mask = (periods > 1.5) & (periods < 30)
ax7.semilogy(periods[mask], psd[mask], color=ACC1, lw=1.5)
ax7.axvspan(2.5, 6, alpha=0.2, color=ACC2, label='Ondas tropicales')
ax7.axvspan(10, 20, alpha=0.15, color=ACC3, label='Variab. intraseasonal')
ax7.set_xlabel('Período (días)'); ax7.set_ylabel('PSD')
ax7.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT)

# ─── Título general
fig.suptitle(
    'Intercambio de Energía: Jet Barotropico del Este ↔ Ondas Tropicales del Este\n'
    'Conversión Barotropica  |  EKE  |  Flujo de Momento  |  Espectro',
    color=TEXT, fontsize=12, fontweight='bold', y=0.98
)

#plt.savefig('/mnt/user-data/outputs/energia_jet_ondas_tropicales.png',
#            dpi=150, bbox_inches='tight', facecolor=DARK)
#print("\n  Figura guardada: energia_jet_ondas_tropicales.png")

# ──────────────────────────────────────────────────────────
# 6.  REPORTE DIAGNÓSTICO
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DIAGNÓSTICO ENERGÉTICO")
print("=" * 60)
print(f"  Criterio de inestabilidad barotropica (Rayleigh-Kuo):")
beta = 2.3e-11  # s⁻¹ m⁻¹
PV_grad = beta - np.gradient(du_dy, np.deg2rad(lats) * 6.371e6)
sign_changes = np.where(np.diff(np.sign(PV_grad)))[0]
if len(sign_changes):
    print(f"  → ∂q/∂y cambia de signo en {len(sign_changes)} lugar(es): condición necesaria CUMPLIDA")
    for idx in sign_changes:
        print(f"     @ {lats[idx]:.1f}°N")
else:
    print("  → No se cumple la condición de cambio de signo de ∂q/∂y")

C_source = lats[C_bt > 0]
print(f"\n  Regiones de generación de EKE (C_BT > 0): {C_source.min():.1f}° – {C_source.max():.1f}°N" if len(C_source) else "  No hay regiones de generación clara")
print(f"  Flujo neto de momento <u'v'>: {np.trapezoid(uv_mean, lats):.3f} m²/s²")
print(f"\n  → Interpretación: el jet del este proporciona energía a las ondas")
print(f"    a través de inestabilidad barotropica donde ∂(ū)/∂y cambia de signo.")
print("=" * 60)

plt.show()
