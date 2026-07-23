"""
Fit N Gaussian components to the zonal-mean profile of the Caribbean
Low-Level Jet (CLLJ, 925 hPa) using the ERA5 JAS climatology
(1991–2020).

The meridional profile is obtained by averaging the zonal wind over
100°W–60°W and is subsequently fitted with models containing 1, 2,
and 3 Gaussian components. The best model is selected using the
corrected Akaike Information Criterion (AICc).

Output:
    Parameters (A_i, phi_i, sigma_i) ready to be used in the
    Jet_Field() function of FluidSim, using centered coordinates
    y = latitude − 15°N.
"""

import os
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0. Configuracion
# ------------------------------------------------------------------
NC_FILE   = "era5_cllj_925.nc"
LEVEL     = "925"                # nucleo del CLLJ
YEARS     = range(1991, 2021)    # climatologia de 30 años
MONTHS    = ["07", "08", "09"]   # JAS -> maximo del CLLJ

AREA      = [25, -120, 5, -60]   # caja de descarga [N, W, S, E]
LON_FIT   = (-100.0, -60.0)      # dominio longitudinal del ajuste
LAT_FIT   = (5.0, 25.0)

LAT0      = 15.0    # centro del dominio -> y = 0 en coordenada centrada
M         = 111e3   # grados -> metros

N_COMPONENTS = [1, 2, 3]   # modelos a comparar


# ------------------------------------------------------------------
# 1. Descarga ERA5
# ------------------------------------------------------------------
def download_era5(path=NC_FILE):
    if os.path.exists(path):
        print(f"[skip] {path} ya existe.")
        return path

    import cdsapi
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": "u_component_of_wind",
            "pressure_level": LEVEL,
            "year": [str(y) for y in YEARS],
            "month": MONTHS,
            "time": "00:00",
            "area": AREA,
            "format": "netcdf",
        },
        path,
    )
    print(f"[ok] descargado -> {path}")
    return path


# ------------------------------------------------------------------
# 2. Perfil climatologico u_bar(lat)
# ------------------------------------------------------------------
def build_profile(path=NC_FILE):
    ds = xr.open_dataset(path)
    u = ds["u"]
    tdim = "valid_time" if "valid_time" in u.dims else "time"

    if float(u.longitude.min()) >= 0:
        u = u.assign_coords(longitude=(((u.longitude + 180) % 360) - 180))
        u = u.sortby("longitude")

    u = u.sel(longitude=slice(*LON_FIT))
    lat_slice = (slice(LAT_FIT[1], LAT_FIT[0])
                 if float(u.latitude[0]) > float(u.latitude[-1])
                 else slice(*LAT_FIT))
    u = u.sel(latitude=lat_slice)

    for lev in ("pressure_level", "level"):
        if lev in u.dims:
            u = u.isel({lev: 0})

    u_bar = u.mean(dim=[tdim, "longitude"])

    lat = u_bar.latitude.values.astype(float)
    ub  = u_bar.values.astype(float)
    idx = np.argsort(lat)
    lat, ub = lat[idx], ub[idx]

    ds.close()
    return lat, ub


# ------------------------------------------------------------------
# 3. Modelo de N gaussianas
# ------------------------------------------------------------------
def n_gaussians(y, *params):
    """params = [A1, p1, s1, A2, p2, s2, ...]. A<0 = easterly."""
    y = np.asarray(y, dtype=float)
    g = np.zeros_like(y)
    for i in range(len(params) // 3):
        A, p, s = params[3 * i: 3 * i + 3]
        g += A * np.exp(-((y - p) ** 2) / (2.0 * s ** 2))
    return g


def initial_guess(n, lat, ub):
    """Semilla: la componente 1 en el minimo observado; resto flanqueando."""
    y = lat - LAT0
    y_min = float(y[np.argmin(ub)])
    A_min = float(np.min(ub))

    if n == 1:
        return [A_min, y_min, 2.5]
    if n == 2:
        return [A_min,       y_min,       1.5,
                0.4 * A_min, y_min + 4.0, 3.0]
    return [A_min,       y_min,       1.5,
            0.5 * A_min, y_min + 4.0, 2.5,
            0.3 * A_min, y_min + 8.0, 3.5]


def fit_n(lat, ub, n):
    """Ajuste de n gaussianas. Devuelve dict con parametros y metricas."""
    y = lat - LAT0
    p0 = initial_guess(n, lat, ub)

    # Amplitudes negativas (easterly); centros dentro del dominio; sigma > 0
    lower, upper = [], []
    for _ in range(n):
        lower += [-25.0, -10.0, 0.3]
        upper += [  0.0,  10.0, 15.0]

    # La semilla debe caer estrictamente dentro de las cotas; recortarla
    # con un margen para evitar que el optimizador arranque en la frontera.
    eps = 1e-6
    p0 = [min(max(v, lo + eps), hi - eps)
          for v, lo, hi in zip(p0, lower, upper)]

    popt, pcov = curve_fit(n_gaussians, y, ub, p0=p0,
                           bounds=(lower, upper), maxfev=400000)

    resid  = ub - n_gaussians(y, *popt)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ub - ub.mean()) ** 2))
    rmse   = float(np.sqrt(ss_res / len(y)))
    r2     = 1.0 - ss_res / ss_tot

    # AIC gaussiano + correccion de muestra pequeña (AICc)
    n_obs = len(y)
    k     = 3 * n + 1                     # +1 por la varianza estimada
    aic   = n_obs * np.log(ss_res / n_obs) + 2 * k
    if n_obs - k - 1 > 0:
        aicc = aic + (2 * k * (k + 1)) / (n_obs - k - 1)
    else:
        aicc = np.inf                     # modelo no identificable

    # pcov puede venir con inf/nan si el ajuste es degenerado.
    with np.errstate(invalid="ignore"):
        perr = np.sqrt(np.diag(pcov))
    perr = np.where(np.isfinite(perr), perr, np.inf)

    # Diagnostico: una componente esta "restringida" si |A| > 2*sigma_A
    ok = [bool(abs(popt[3 * i]) > 2.0 * perr[3 * i]) for i in range(n)]

    return dict(n=n, popt=popt, perr=perr, rmse=rmse, r2=r2,
                aicc=aicc, k=k, n_obs=n_obs, well_constrained=ok, y=y)


# ------------------------------------------------------------------
# 4. Reportes
# ------------------------------------------------------------------
def report_one(fit):
    n, popt, perr = fit["n"], fit["popt"], fit["perr"]
    print(f"\n--- {n} gaussiana(s) "
          f"| RMSE {fit['rmse']:.3f} m/s "
          f"| R2 {fit['r2']:.4f} "
          f"| AICc {fit['aicc']:.2f} ---")
    print(f"{'i':>2} {'A_i [m/s]':>18} {'phi_i [deg y]':>18} "
          f"{'lat real':>10} {'sigma_i [deg]':>18}  ok")
    for i in range(n):
        A, p, s    = popt[3 * i: 3 * i + 3]
        dA, dp, ds = perr[3 * i: 3 * i + 3]
        flag = "si" if fit["well_constrained"][i] else "NO"
        print(f"{i+1:>2} {A:>9.3f}+-{dA:<7.3f} {p:>9.3f}+-{dp:<7.3f} "
              f"{p + LAT0:>9.2f}N {s:>9.3f}+-{ds:<7.3f}  {flag}")


def report_comparison(fits):
    print("\n" + "=" * 70)
    print("COMPARACION DE MODELOS  |  CLLJ 925 hPa  |  JAS  |  100W-60W")
    print("=" * 70)
    for f in fits:
        report_one(f)

    best = min(fits, key=lambda f: f["aicc"])
    aicc_min = best["aicc"]

    print("\n" + "-" * 70)
    print(f"{'modelo':>10} {'k':>4} {'RMSE':>9} {'R2':>9} "
          f"{'AICc':>10} {'dAICc':>9} {'restringidas':>14}")
    print("-" * 70)
    for f in fits:
        d = f["aicc"] - aicc_min
        n_ok = sum(f["well_constrained"])
        mark = "  <-- mejor" if f is best else ""
        print(f"{f['n']:>7} gauss {f['k']:>4} {f['rmse']:>9.3f} "
              f"{f['r2']:>9.4f} {f['aicc']:>10.2f} {d:>9.2f} "
              f"{n_ok:>8}/{f['n']:<5}{mark}")

    print("\nCriterio: menor AICc gana. dAICc < 2 -> modelos equivalentes;")
    print("dAICc > 10 -> el modelo peor queda descartado.")
    print("'restringidas' = componentes con |A| > 2*sigma_A (senal > 2-sigma).")
    return best


def report_fluidsim(fit):
    n, popt = fit["n"], fit["popt"]
    print("\n" + "-" * 70)
    print(f"Bloque para Jet_Field()  [modelo seleccionado: {n} gaussiana(s)]")
    print("-" * 70)
    print("    for lat, amp, sigma in [")
    order = sorted(range(n), key=lambda i: -abs(popt[3 * i]))
    for i in order:
        A, p, s = popt[3 * i: 3 * i + 3]
        print(f"        ({p:>7.3f}*m, {A:>8.3f}, {s:>6.3f}*m),"
              f"   # lat real {p + LAT0:.2f}N")
    print("    ]:")
    print("        u_bar += amp * np.exp(-((lats - lat)**2 / (2*sigma**2)))")

    # Diagnosticos fisicos del nucleo (componente de mayor |A|)
    i0 = max(range(n), key=lambda i: abs(popt[3 * i]))
    A, p, s    = popt[3 * i0: 3 * i0 + 3]
    dA, dp, ds = fit["perr"][3 * i0: 3 * i0 + 3]
    fwhm  = 2.0 * np.sqrt(2.0 * np.log(2.0)) * s
    dfwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * ds

    print("\n" + "-" * 70)
    print("Nucleo del CLLJ (componente dominante):")
    print("-" * 70)
    print(f"  Latitud       : {p + LAT0:.2f} +- {dp:.2f} N   (y = {p:+.2f} deg)")
    print(f"  Intensidad    : {A:.2f} +- {dA:.2f} m/s")
    print(f"  sigma         : {s:.2f} +- {ds:.2f} deg")
    print(f"  FWHM (2.355s) : {fwhm:.2f} +- {dfwhm:.2f} deg")

    off = abs(p)
    if off > 1.0:
        print(f"\n  [aviso] el nucleo cae {off:.2f} deg fuera del centro de la")
        print(f"          grilla (y=0 <-> {LAT0:.0f}N). Considerar recentrar el")
        print(f"          dominio en {p + LAT0:.1f}N, o verificar estabilidad")
        print(f"          numerica con el jet descentrado.")


# ------------------------------------------------------------------
# 5. Figura
# ------------------------------------------------------------------
def plot_fits(lat, ub, fits, best, path="cllj_fit.png"):
    y = lat - LAT0
    y_fine = np.linspace(y.min(), y.max(), 400)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # Panel 0: todos los modelos
    ax[0].plot(ub, lat, "ko", ms=4, label="ERA5 (JAS clim.)")
    for f in fits:
        ls  = "-" if f is best else "--"
        lw  = 2.2 if f is best else 1.2
        lbl = f"{f['n']}G (AICc {f['aicc']:.1f})"
        if f is best:
            lbl += "  *"
        ax[0].plot(n_gaussians(y_fine, *f["popt"]), y_fine + LAT0,
                   ls, lw=lw, alpha=0.9, label=lbl)
    ax[0].axvline(0, color="gray", ls=":", lw=1)
    ax[0].set_xlabel(r"$\bar{u}$ [m/s]")
    ax[0].set_ylabel("Latitud [N]")
    ax[0].set_title("Comparacion de modelos")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # Panel 1: descomposicion del mejor
    ax[1].plot(ub, lat, "ko", ms=4, label="ERA5")
    ax[1].plot(n_gaussians(y_fine, *best["popt"]), y_fine + LAT0,
               "r-", lw=2.2, label=f"Ajuste {best['n']}G")
    for i in range(best["n"]):
        A, p, s = best["popt"][3 * i: 3 * i + 3]
        ax[1].plot(A * np.exp(-((y_fine - p) ** 2) / (2 * s ** 2)),
                   y_fine + LAT0, "--", lw=1, alpha=0.7, label=f"G{i+1}")
    ax[1].axhline(LAT0, color="gray", ls=":", lw=1)
    ax[1].axvline(0, color="gray", ls=":", lw=1)
    ax[1].set_xlabel(r"$\bar{u}$ [m/s]")
    ax[1].set_title(f"Modelo seleccionado ({best['n']} gaussianas)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    # Panel 2: residuos
    for f in fits:
        resid = ub - n_gaussians(y, *f["popt"])
        lw = 2.0 if f is best else 1.0
        ax[2].plot(resid, lat, ".-", ms=4, lw=lw,
                   alpha=0.9 if f is best else 0.5,
                   label=f"{f['n']}G  (RMSE {f['rmse']:.2f})")
    ax[2].axvline(0, color="k", lw=1)
    ax[2].set_xlabel("Residuo [m/s]")
    ax[2].set_title("Residuos")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\n[ok] figura -> {path}")


# ------------------------------------------------------------------
if __name__ == "__main__":
    download_era5()
    lat, ub = build_profile()

    fits = []
    for n in N_COMPONENTS:
        try:
            fits.append(fit_n(lat, ub, n))
        except RuntimeError as e:
            print(f"[warn] el ajuste de {n} gaussianas no convergio: {e}")

    best = report_comparison(fits)
    report_fluidsim(best)
    plot_fits(lat, ub, fits, best)
