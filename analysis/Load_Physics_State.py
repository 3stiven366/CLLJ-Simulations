"""
Load ALL state_phys_t*.nc files into memory: x, y, times, ux, uy, rot.

Intended to run on the cluster (plenty of RAM available), not in streaming mode.
Each .nc file contains:
    state_phys: y, x, ux, uy, rot
    info_simul: params, solver   (redundant across files -> read only once)
"""

import os
import csv
import json
import glob
import re
import xml.etree.ElementTree as ET
import numpy as np
import h5py


def _list_files(directory, pattern="state_phys_t*.nc"):
    """List files sorted chronologically, extracting t from the filename."""
    files = glob.glob(f"{directory}/{pattern}")
    if not files:
        raise FileNotFoundError(f"No files found matching pattern {pattern} in {directory}")

    def extract_t(name):
        m = re.search(r"_t(\d+\.\d+)", name)
        return float(m.group(1))

    sorted_files = sorted(files, key=extract_t)
    times = np.array([extract_t(f) for f in sorted_files])
    return sorted_files, times


def _build_coords(directory, ny, nx, x_file, y_file):
    """
    Devuelve (x, y) físicas. Los .nc de FluidSim con MPI a veces guardan las
    coordenadas x/y en CERO (o con tamaño incorrecto); en ese caso se
    reconstruyen desde Lx, Ly de params_simul.xml, asumiendo un dominio
    periódico uniforme (0..L, sin incluir el extremo -> endpoint=False).
    Si las coordenadas del archivo ya vienen bien, se conservan tal cual.
    """
    def read_LxLy():
        path = os.path.join(directory, "params_simul.xml")
        if not os.path.exists(path):
            return None, None
        oper = ET.parse(path).getroot().find("oper")
        if oper is None:
            return None, None
        return float(oper.attrib["Lx"]), float(oper.attrib["Ly"])

    x, y = x_file, y_file
    need_x = np.all(x_file == 0) or x_file.size != nx
    need_y = np.all(y_file == 0) or y_file.size != ny
    if need_x or need_y:
        Lx, Ly = read_LxLy()
        if need_x:
            if Lx is None:
                raise ValueError("Coordenada x trivial y no hay params_simul.xml para reconstruir Lx")
            x = np.linspace(0, Lx, nx, endpoint=False)
        if need_y:
            if Ly is None:
                raise ValueError("Coordenada y trivial y no hay params_simul.xml para reconstruir Ly")
            y = np.linspace(0, Ly, ny, endpoint=False)
        print("[coords] coordenadas triviales en el .nc; reconstruidas desde params_simul.xml")
    return x, y


def estimate_memory_gb(directory, pattern="state_phys_t*.nc", n_fields=3):
    """
    Quick estimate of how much RAM you'll need BEFORE loading everything.
    Useful to decide whether it fits in the cluster node you were allocated.
    """
    files, times = _list_files(directory, pattern)
    with h5py.File(files[0], "r") as f0:
        ny = f0["state_phys"]["y"].shape[0]
        nx = f0["state_phys"]["x"].shape[0]

    nt = len(files)
    total_bytes = nt * ny * nx * n_fields * 8  # float64
    gb = total_bytes / 1024**3
    print(f"nt={nt}, ny={ny}, nx={nx}, fields={n_fields} -> ~{gb:.2f} GB")
    return gb


def load_full_simulation(directory, pattern="state_phys_t*.nc", dtype=np.float64):
    """
    Returns:
        x        : array (nx,)      -- zonal coordinates (assumed identical across files)
        y        : array (ny,)      -- meridional coordinates
        times    : array (nt,)      -- time of each snapshot, in the order of the returned arrays
        ux, uy, rot : arrays (nt, ny, nx) -- full fields for all times
    """
    files, times = _list_files(directory, pattern)
    nt = len(files)

    with h5py.File(files[0], "r") as f0:
        ny, nx = f0["state_phys"]["ux"].shape   # forma real del campo
        x_file = f0["state_phys"]["x"][:]
        y_file = f0["state_phys"]["y"][:]

    # reconstruye x/y si vienen triviales (todas cero) en el .nc
    x, y = _build_coords(directory, ny, nx, x_file, y_file)

    ux = np.empty((nt, ny, nx), dtype=dtype)
    uy = np.empty((nt, ny, nx), dtype=dtype)
    rot = np.empty((nt, ny, nx), dtype=dtype)

    for i, file in enumerate(files):
        with h5py.File(file, "r") as f:
            ux[i] = f["state_phys"]["ux"][:]
            uy[i] = f["state_phys"]["uy"][:]
            rot[i] = f["state_phys"]["rot"][:]

        if (i + 1) % 50 == 0 or (i + 1) == nt:
            print(f"  loaded {i + 1}/{nt} files", flush=True)

    return x, y, times, ux, uy, rot


# ===============================================================
# Lectores de los diagnósticos que FluidSim ya calculó
# Todos reciben el directorio de la corrida (path_run) y buscan el
# archivo por su nombre estándar. Devuelven dict de arrays de numpy.
# ===============================================================

def load_spatial_means(directory, filename="spatial_means.txt"):
    """
    Series temporales globales: energía E, enstrofía Z, disipaciones
    (epsK, epsZ) e inyecciones del forzamiento (PK, PZ).

    Devuelve dict con: time, E, Z, epsK, epsK_hypo, epsK_tot, epsZ,
    epsZ_hypo, epsZ_tot, PK1, PK2, PK_tot, PZ1, PZ2, PZ_tot  (arrays (nt,)).
    """
    path = os.path.join(directory, filename)
    pattern = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([-+0-9.eE]+)")
    data = {}
    with open(path) as f:
        for line in f:
            for key, val in pattern.findall(line):
                data.setdefault(key, []).append(float(val))
    return {k: np.array(v) for k, v in data.items()}


def load_jet_diagnostics(directory, filename="jet_diagnostics.csv"):
    """
    Diagnósticos del jet calculados en la corrida: amplitud relativa del jet
    (A_jet), energía cinética zonal vs. eddy (KE_zonal, KE_eddy), tasa de
    nudging (F_jet_rate), y qué eventos de onda estaban activos.

    Devuelve dict de arrays. Si el CSV solo tiene encabezado (corrida
    cancelada antes de escribir datos), devuelve arrays vacíos y avisa.
    """
    path = os.path.join(directory, filename)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    if not rows:
        print(f"[aviso] {filename} solo tiene encabezado, sin datos.")
        return {col: np.array([]) for col in header}
    cols = list(zip(*rows))
    return {name: np.array(col, dtype=float) for name, col in zip(header, cols)}


def load_wave_catalog(directory, filename="wave_catalog.csv"):
    """
    Catálogo de eventos de onda inyectados como forzamiento (es el INPUT
    documentado, no un resultado): longitud de onda, período, velocidad de
    fase, amplitud, latitud central, etc., por evento.

    Devuelve dict de arrays (una entrada por columna).
    """
    path = os.path.join(directory, filename)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    if not rows:
        return {col: np.array([]) for col in header}
    cols = list(zip(*rows))
    return {name: np.array(col, dtype=float) for name, col in zip(header, cols)}


def load_spectra1D(directory, filename="spectra1D.h5"):
    """
    Espectros 1D de energía en kx y ky por separado, por tiempo.
    Devuelve dict con:
        times : (nt,)
        kx    : (nkx,)   números de onda zonales
        ky    : (nky,)   números de onda meridionales
        Ekx   : (nt, nkx) espectro E(kx, t)
        Eky   : (nt, nky) espectro E(ky, t)
    """
    path = os.path.join(directory, filename)
    with h5py.File(path, "r") as f:
        return {
            "times": f["times"][:],
            "kx": f["kxE"][:],
            "ky": f["kyE"][:],
            "Ekx": f["spectrum1Dkx_E"][:],
            "Eky": f["spectrum1Dky_E"][:],
        }


def load_spectra2D(directory, filename="spectra2D.h5"):
    """
    Espectro isótropo de energía vs. número de onda total kh, por tiempo.
    Devuelve dict con: times (nt,), kh (nkh,), E2D (nt, nkh).
    """
    path = os.path.join(directory, filename)
    with h5py.File(path, "r") as f:
        return {
            "times": f["times"][:],
            "kh": f["khE"][:],
            "E2D": f["spectrum2D_E"][:],
        }


def load_spect_energy_budg(directory, filename="spect_energy_budg.h5"):
    """
    Presupuesto de energía y enstrofía EN EL ESPACIO ESPECTRAL: transferencia
    por número de onda kh, por tiempo. Dice a qué escalas se inyecta/extrae
    energía (complementa a la CBT, que lo dice por latitud).

    Devuelve dict con:
        times      : (nt,)
        kh         : (nkh,)
        transfer_E : (nt, nkh)  transferencia espectral de energía
        transfer_Z : (nt, nkh)  transferencia espectral de enstrofía
    """
    path = os.path.join(directory, filename)
    with h5py.File(path, "r") as f:
        return {
            "times": f["times"][:],
            "kh": f["khE"][:],
            "transfer_E": f["transfer2D_E"][:],
            "transfer_Z": f["transfer2D_Z"][:],
        }


def load_increments(directory, filename="increments.h5"):
    """
    PDFs de incrementos de velocidad/vorticidad y extremos por escala
    (diagnóstico de intermitencia de la turbulencia 2D).

    Devuelve dict con: times, rxs (separaciones), nbins, y los
    pdf_delta_* / valmax_* / valmin_* disponibles en el archivo.
    """
    path = os.path.join(directory, filename)
    out = {}
    with h5py.File(path, "r") as f:
        out["times"] = f["times"][:]
        out["rxs"] = f["rxs"][:]
        out["nbins"] = int(f["nbins"][()])
        for key in ["pdf_delta_ux", "pdf_delta_uy", "pdf_delta_rot",
                    "valmax_ux", "valmax_uy", "valmax_rot",
                    "valmin_ux", "valmin_uy", "valmin_rot"]:
            if key in f:
                out[key] = f[key][:]
    return out


# ===============================================================
# Configuración de la corrida
#   - load_config    : constantes propias de CLLJ_simulation.py (JSON) -> flags,
#                      parámetros del jet, rangos de onda, etc.
#   - load_params_xml: config que FluidSim guarda solo (params_simul.xml)
# ===============================================================

def load_config(directory, filename="config_experimento.json"):
    """
    Lee config_experimento.json (las constantes propias del experimento que
    FluidSim no registra: ACTIVE_WAVES, ACTIVE_JET_NUDGING, JET_PARAMS, ...).
    Devuelve dict, o None si el archivo no existe (corridas viejas sin él).
    """
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_params_xml(directory, filename="params_simul.xml"):
    """
    Lee params_simul.xml (config que FluidSim vuelca automáticamente).
    Devuelve dict con los parámetros clave, tipados. None si no existe.
    """
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        return None
    root = ET.parse(path).getroot()

    def get(elem, key, cast=float, default=None):
        if elem is None or key not in elem.attrib:
            return default
        try:
            return cast(elem.attrib[key])
        except (ValueError, TypeError):
            return elem.attrib[key]   # deja el string si no castea (p.ej. 'None')

    oper = root.find("oper")
    ts   = root.find("time_stepping")
    forc = root.find("forcing")

    return {
        "beta": get(root, "beta"),
        "nu_2": get(root, "nu_2"),
        "nu_4": get(root, "nu_4"),
        "path_run": root.attrib.get("path_run", ""),
        "Lx": get(oper, "Lx"),
        "Ly": get(oper, "Ly"),
        "nx": get(oper, "nx", int),
        "ny": get(oper, "ny", int),
        "type_fft": oper.attrib.get("type_fft", "") if oper is not None else "",
        "t_end": get(ts, "t_end"),
        "deltat_max": get(ts, "deltat_max"),
        "type_time_scheme": ts.attrib.get("type_time_scheme", "") if ts is not None else "",
        "forcing_enable": forc.attrib.get("enable", "") if forc is not None else "",
        "forcing_type": forc.attrib.get("type", "") if forc is not None else "",
        "key_forced": forc.attrib.get("key_forced", "") if forc is not None else "",
    }


if __name__ == "__main__":
    directory = "."  # folder with your state_phys_t*.nc files

    # 1) Check how much RAM you'll need before loading everything
    estimate_memory_gb(directory)

    # 2) Full load
    x, y, times, ux, uy, rot = load_full_simulation(directory)

    print("x:", x.shape, "y:", y.shape)
    print("times:", times.shape, times.min(), "->", times.max())
    print("ux/uy/rot:", ux.shape)  # (nt, ny, nx)
