#!/usr/bin/env bash
# ============================================================================
#  AgriBot installer for NVIDIA Jetson Orin Nano (JetPack 6.x)
#
#    sudo ./deploy/install_jetson.sh
#
#  Installs the stack to /opt/agribot in a virtualenv, sets up the service
#  user, udev rules and the systemd unit, and runs the preflight checks.
#
#  It does NOT install torch or ultralytics. On Jetson those must come from
#  NVIDIA's own wheel index, not PyPI - a pip-installed torch is a CPU build
#  and silently gives up the GPU the whole two-tier design depends on. See
#  the note printed at the end.
#
#  Safe to re-run: every step is idempotent.
# ============================================================================
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/agribot}"
SERVICE_USER="${SERVICE_USER:-agribot}"
PYTHON="${PYTHON:-python3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()   { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run with sudo"

# ---------------------------------------------------------------------------
info "Checking the platform"
# ---------------------------------------------------------------------------
if [[ -r /proc/device-tree/model ]]; then
    MODEL="$(tr -d '\0' < /proc/device-tree/model)"
    info "  detected: ${MODEL}"
    case "${MODEL,,}" in
        *orin*|*jetson*|*tegra*) ;;
        *) warn "  not a Jetson - installing anyway, but TensorRT export will not work" ;;
    esac
else
    warn "  cannot read the device tree; assuming a generic Linux host"
fi

command -v "${PYTHON}" >/dev/null || die "${PYTHON} not found"
PY_VER="$(${PYTHON} -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
info "  python ${PY_VER}"
${PYTHON} -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' \
    || die "Python 3.8+ required"

# ---------------------------------------------------------------------------
info "Installing system packages"
# ---------------------------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    python3-venv python3-dev python3-pip \
    libopencv-dev python3-opencv \
    v4l-utils i2c-tools \
    build-essential git

# ---------------------------------------------------------------------------
info "Creating the service user"
# ---------------------------------------------------------------------------
if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
    useradd --system --create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
    info "  created ${SERVICE_USER}"
else
    info "  ${SERVICE_USER} already exists"
fi
# dialout for the MCU serial link, video for the camera, i2c for the OLED.
usermod -aG dialout,video,i2c "${SERVICE_USER}" 2>/dev/null || \
    usermod -aG dialout,video "${SERVICE_USER}"

# ---------------------------------------------------------------------------
info "Installing the application to ${INSTALL_DIR}"
# ---------------------------------------------------------------------------
mkdir -p "${INSTALL_DIR}"
for item in src config tools firmware deploy docs pyproject.toml requirements.txt \
            requirements-sim.txt requirements-ml.txt README.md; do
    [[ -e "${REPO_DIR}/${item}" ]] && cp -r "${REPO_DIR}/${item}" "${INSTALL_DIR}/"
done
mkdir -p "${INSTALL_DIR}/data/logs" "${INSTALL_DIR}/models"

# ---------------------------------------------------------------------------
info "Building the virtualenv"
# ---------------------------------------------------------------------------
if [[ ! -d "${INSTALL_DIR}/.venv" ]]; then
    # --system-site-packages so the JetPack-provided OpenCV (built with CUDA)
    # is visible instead of a pip wheel that is not.
    ${PYTHON} -m venv --system-site-packages "${INSTALL_DIR}/.venv"
fi
"${INSTALL_DIR}/.venv/bin/pip" install --quiet --upgrade pip wheel
"${INSTALL_DIR}/.venv/bin/pip" install --quiet -e "${INSTALL_DIR}"
info "  installed agribot and its runtime dependencies"

# ---------------------------------------------------------------------------
info "Installing the udev rule for the MCU"
# ---------------------------------------------------------------------------
# A stable symlink means the config never has to chase /dev/ttyACM0 vs ACM1
# after a re-plug, which is exactly the kind of thing that eats arena time.
cat > /etc/udev/rules.d/99-agribot.rules <<'RULES'
# Arduino Uno / Mega on the AgriBot custom PCB -> /dev/agribot-mcu
SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", MODE="0660", GROUP="dialout", SYMLINK+="agribot-mcu"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", MODE="0660", GROUP="dialout", SYMLINK+="agribot-mcu"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", MODE="0660", GROUP="dialout", SYMLINK+="agribot-mcu"
RULES
udevadm control --reload-rules && udevadm trigger || warn "  udev reload failed"
info "  MCU will appear as /dev/agribot-mcu"

# ---------------------------------------------------------------------------
info "Installing the systemd unit"
# ---------------------------------------------------------------------------
cp "${REPO_DIR}/deploy/agribot.service" /etc/systemd/system/
systemctl daemon-reload
info "  installed (NOT enabled - start it deliberately)"

chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_DIR}"

# ---------------------------------------------------------------------------
info "Setting the Jetson to maximum performance"
# ---------------------------------------------------------------------------
if command -v nvpmodel >/dev/null; then
    nvpmodel -m 0 || warn "  nvpmodel failed"
fi
if command -v jetson_clocks >/dev/null; then
    jetson_clocks || warn "  jetson_clocks failed"
fi

# ---------------------------------------------------------------------------
info "Running preflight (config and detector checks only)"
# ---------------------------------------------------------------------------
if sudo -u "${SERVICE_USER}" "${INSTALL_DIR}/.venv/bin/python" \
        -m agribot.app.preflight --skip-hardware; then
    info "  preflight passed"
else
    warn "  preflight reported problems - fix them before running the mission"
fi

cat <<EOF

============================================================================
  AgriBot installed to ${INSTALL_DIR}

  Next steps
  ----------
  1. Flash the MCU firmware:
       arduino-cli compile --fqbn arduino:avr:mega ${INSTALL_DIR}/firmware/agribot_mcu
       arduino-cli upload  --fqbn arduino:avr:mega -p /dev/agribot-mcu \\
                           ${INSTALL_DIR}/firmware/agribot_mcu

  2. Point the config at the stable serial symlink:
       sudo -u ${SERVICE_USER} sed -i 's|/dev/ttyACM0|/dev/agribot-mcu|' \\
            ${INSTALL_DIR}/config/robot.yaml

  3. Full preflight with hardware attached:
       sudo -u ${SERVICE_USER} ${INSTALL_DIR}/.venv/bin/python -m agribot.app.preflight

  4. Calibrate on the arena surface (see docs/CALIBRATION.md):
       ${INSTALL_DIR}/.venv/bin/python tools/calibrate_hsv.py --target line
       ${INSTALL_DIR}/.venv/bin/python tools/calibrate_spray.py

  5. Dry run first - navigation and perception, valve inhibited:
       sudo -u ${SERVICE_USER} ${INSTALL_DIR}/.venv/bin/python \\
            -m agribot.app.main --dry-run

  6. Then the real thing:
       sudo systemctl start agribot
       journalctl -u agribot -f

  Optional: the learned detector (Tier 2)
  ---------------------------------------
  The colour tier is the guaranteed scorer and runs with no extra install.
  To add the learned tier, install torch/torchvision from NVIDIA's Jetson
  index FIRST - a PyPI torch is a CPU build and gives up the GPU entirely:

       ${INSTALL_DIR}/.venv/bin/pip install --no-cache \\
           --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 torch torchvision
       ${INSTALL_DIR}/.venv/bin/pip install ultralytics
       ${INSTALL_DIR}/.venv/bin/python tools/export_tensorrt.py --weights best.pt

  Then set perception.yolo.enabled: true in config/robot.yaml.
============================================================================
EOF
