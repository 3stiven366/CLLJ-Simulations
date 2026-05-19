#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# install_fluidsim.sh
#
# Instala FluidSim con soporte MPI en Fedora usando conda.
# Corre en el environment conda que tengas activo al momento de ejecutarlo.
#
# Uso:
#   conda activate mi_env   # (opcional, si no querés usar base)
#   bash install_fluidsim.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail  # salir si cualquier comando falla

# ── Colores para mensajes ────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # sin color

info()    { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }
section() { echo -e "\n${BLUE}━━━ $1 ━━━${NC}"; }

# ─────────────────────────────────────────────────────────────────────────────
# 0. VERIFICACIONES PREVIAS
# ─────────────────────────────────────────────────────────────────────────────
section "Verificaciones previas"

# Verificar que estamos en Fedora/RHEL
if ! command -v dnf &>/dev/null; then
    error "Este script requiere dnf (Fedora/RHEL). No se encontró dnf."
    exit 1
fi
ok "Sistema compatible (dnf encontrado)"

# Verificar que conda está disponible
if ! command -v conda &>/dev/null; then
    error "conda no está disponible. Activá conda antes de correr este script."
    error "Ejemplo: source ~/miniforge3/etc/profile.d/conda.sh"
    exit 1
fi
ok "conda disponible: $(conda --version)"

# Mostrar el environment activo
CONDA_ENV="${CONDA_DEFAULT_ENV:-base}"
info "Environment conda activo: ${CONDA_ENV}"
info "Python activo: $(which python3) — $(python3 --version)"

# Verificar sudo
if ! sudo -n true 2>/dev/null; then
    warn "Se necesita sudo para instalar paquetes del sistema. Se te pedirá la contraseña."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1. PYTHON (via conda)
# ─────────────────────────────────────────────────────────────────────────────
section "Python"

info "Instalando/actualizando Python en el environment activo..."
conda install -y python --channel conda-forge 2>/dev/null || true
ok "Python: $(python3 --version)"

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEPENDENCIAS DEL SISTEMA (dnf)
# ─────────────────────────────────────────────────────────────────────────────
section "Dependencias del sistema (dnf)"

SYSTEM_PKGS=(
    gcc          # compilador C
    gcc-c++      # compilador C++ (necesario para Cython/Meson)
    make
    python3-devel  # headers de Python para compilar extensiones C
    fftw-devel     # FFTW3 (transformadas de Fourier rápidas)
    fftw-libs      # librerías runtime de FFTW
    openmpi        # runtime de OpenMPI
    openmpi-devel  # headers de OpenMPI para compilar
    hdf5-devel     # para I/O de fluidsim
    h5py           # binding Python de HDF5 (dnf para evitar problemas de compilación)
)

info "Instalando paquetes del sistema..."
sudo dnf install -y "${SYSTEM_PKGS[@]}"
ok "Paquetes del sistema instalados"

# ─────────────────────────────────────────────────────────────────────────────
# 3. MÓDULO MPI
# ─────────────────────────────────────────────────────────────────────────────
section "Configuración de OpenMPI"

# Cargar el módulo MPI para esta sesión
MPI_MODULE="mpi/openmpi-x86_64"
if command -v module &>/dev/null; then
    module load "${MPI_MODULE}" 2>/dev/null && ok "Módulo ${MPI_MODULE} cargado" \
        || warn "No se pudo cargar el módulo ${MPI_MODULE} — continuando de todas formas"
else
    warn "El comando 'module' no está disponible. Intentando con PATH directo."
fi

# Buscar mpicc en rutas comunes de Fedora
MPICC_PATHS=(
    "/usr/lib64/openmpi/bin/mpicc"
    "/usr/lib/openmpi/bin/mpicc"
    "$(which mpicc 2>/dev/null || echo '')"
)

MPICC_FOUND=""
for p in "${MPICC_PATHS[@]}"; do
    if [[ -x "$p" ]]; then
        MPICC_FOUND="$p"
        break
    fi
done

if [[ -n "$MPICC_FOUND" ]]; then
    export MPICC="$MPICC_FOUND"
    # Agregar el bin de OpenMPI al PATH si no está
    MPI_BIN_DIR="$(dirname "$MPICC_FOUND")"
    export PATH="${MPI_BIN_DIR}:${PATH}"
    ok "mpicc encontrado: ${MPICC_FOUND}"
else
    warn "mpicc no encontrado. La instalación de mpi4py puede fallar."
    warn "Intentá: sudo dnf install openmpi-devel && module load mpi/openmpi-x86_64"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. PAQUETES PYTHON VIA CONDA-FORGE
# ─────────────────────────────────────────────────────────────────────────────
section "Paquetes Python (conda-forge)"

# Primero intentar instalar desde conda-forge — tienen builds precompilados
# que evitan los problemas de compilación con Cython/Meson
CONDA_PKGS=(
    numpy
    scipy
    matplotlib
    h5py
    mpi4py          # binding MPI para Python
    cython          # necesario para compilar fluidfft-fftw
    meson           # sistema de build
    meson-python    # plugin de meson para Python
    ninja           # backend de compilación para meson
    pyfftw          # binding Python de FFTW
)

info "Instalando paquetes base desde conda-forge..."
conda install -y "${CONDA_PKGS[@]}" --channel conda-forge
ok "Paquetes base instalados"

# ─────────────────────────────────────────────────────────────────────────────
# 5. FLUIDDYN Y FLUIDSIM
# ─────────────────────────────────────────────────────────────────────────────
section "FluidDyn y FluidSim"

# Intentar conda-forge primero
info "Intentando instalar fluiddyn y fluidsim desde conda-forge..."
if conda install -y fluiddyn fluidsim --channel conda-forge 2>/dev/null; then
    ok "fluiddyn y fluidsim instalados desde conda-forge"
else
    warn "No disponibles en conda-forge — instalando con pip..."
    pip install fluiddyn fluidsim
    ok "fluiddyn y fluidsim instalados con pip"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. FLUIDFFT Y PLUGIN MPI
# ─────────────────────────────────────────────────────────────────────────────
section "FluidFFT con soporte MPI"

# fluidfft base
info "Instalando fluidfft..."
if conda install -y fluidfft --channel conda-forge 2>/dev/null; then
    ok "fluidfft instalado desde conda-forge"
else
    pip install fluidfft
    ok "fluidfft instalado con pip"
fi

# Plugin MPI con FFTW — este requiere compilación
# Usamos el compilador de conda para evitar conflictos con python3-devel del sistema
info "Instalando fluidfft-fftw (requiere compilación, puede tardar)..."

# Apuntar a los headers de Python del environment conda activo
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base)}"
export CFLAGS="-I${CONDA_PREFIX}/include"
export LDFLAGS="-L${CONDA_PREFIX}/lib"

if pip install fluidfft-fftw --no-build-isolation 2>/dev/null; then
    ok "fluidfft-fftw instalado correctamente"
else
    warn "fluidfft-fftw falló. Intentando sin --no-build-isolation..."
    if pip install fluidfft-fftw 2>/dev/null; then
        ok "fluidfft-fftw instalado"
    else
        warn "fluidfft-fftw no se pudo instalar."
        warn "MPI con FFTW no estará disponible, pero podés usar with_pyfftw."
        warn "Para intentarlo manualmente:"
        warn "  module load mpi/openmpi-x86_64"
        warn "  MPICC=mpicc pip install fluidfft-fftw"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 7. AGREGAR MPI AL PATH DE FORMA PERMANENTE
# ─────────────────────────────────────────────────────────────────────────────
section "Configuración permanente de PATH"

RC_FILE="${HOME}/.bashrc"

MPI_LINE='export PATH="/usr/lib64/openmpi/bin:$PATH"'
MPI_LIB_LINE='export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH"'

if ! grep -qF "openmpi/bin" "${RC_FILE}" 2>/dev/null; then
    echo "" >> "${RC_FILE}"
    echo "# OpenMPI — agregado por install_fluidsim.sh" >> "${RC_FILE}"
    echo "${MPI_LINE}" >> "${RC_FILE}"
    echo "${MPI_LIB_LINE}" >> "${RC_FILE}"
    ok "PATH de OpenMPI agregado a ${RC_FILE}"
else
    ok "OpenMPI ya estaba en ${RC_FILE}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 8. VERIFICACIÓN FINAL
# ─────────────────────────────────────────────────────────────────────────────
section "Verificación final"

ERRORS=0

check_import() {
    local pkg="$1"
    local label="${2:-$1}"
    if python3 -c "import ${pkg}" 2>/dev/null; then
        ok "${label}"
    else
        error "${label} — NO se pudo importar"
        ERRORS=$((ERRORS + 1))
    fi
}

check_import "numpy"        "numpy"
check_import "scipy"        "scipy"
check_import "matplotlib"   "matplotlib"
check_import "h5py"         "h5py"
check_import "mpi4py"       "mpi4py (MPI para Python)"
check_import "pyfftw"       "pyfftw (FFTW binding)"
check_import "fluiddyn"     "fluiddyn"
check_import "fluidsim"     "fluidsim"
check_import "fluidfft"     "fluidfft"

# Verificar plugins de fluidfft disponibles
echo ""
info "Plugins de fluidfft disponibles:"
python3 -c "
import fluidfft
plugins = fluidfft.get_plugins()
for p in plugins:
    print(f'    - {p.name}')
" 2>/dev/null || warn "No se pudieron listar los plugins de fluidfft"

# Verificar mpirun
echo ""
if command -v mpirun &>/dev/null; then
    ok "mpirun disponible: $(mpirun --version 2>&1 | head -1)"
else
    error "mpirun no encontrado en PATH"
    ERRORS=$((ERRORS + 1))
fi

# Verificar plugin MPI específico
echo ""
if python3 -c "import fluidfft; [p for p in fluidfft.get_plugins() if 'mpi' in p.name]" 2>/dev/null | grep -q "mpi"; then
    ok "Plugin MPI de fluidfft disponible (fft2d.mpi_with_fftw1d)"
else
    warn "Plugin MPI de fluidfft NO disponible"
    warn "Podés correr sin MPI usando: FLUIDSIM_FFT=fft2d.with_pyfftw python script.py"
fi

# ── Resumen ──────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ $ERRORS -eq 0 ]]; then
    ok "Instalación completada sin errores."
    echo ""
    info "Para correr tu simulación:"
    echo "    sin MPI:  python FluidSim_Barotropic_MPI.py"
    echo "    con MPI:  mpirun -n 4 python FluidSim_Barotropic_MPI.py"
else
    warn "Instalación completada con ${ERRORS} error(es)."
    warn "Revisá los mensajes [ERROR] arriba antes de correr la simulación."
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Recordar hacer source del bashrc
echo ""
info "Recordá recargar tu terminal o correr:"
echo "    source ~/.bashrc"
