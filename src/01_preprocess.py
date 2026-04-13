# src/01_preprocess.py
import numpy as np
from pathlib import Path
from scipy.signal import detrend, butter, filtfilt

try:
    from obspy import read as obspy_read
except Exception:
    obspy_read = None

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _bandpass(x: np.ndarray, fs: float, fmin: float = 0.1, fmax: float = 25.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low = max(1e-4, fmin / nyq)
    high = min(0.99, fmax / nyq)
    if low >= high:
        return x
    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, x)

def _load_obspy(path: Path):
    st = obspy_read(str(path))
    st.merge(fill_value='interpolate')
    tr = st[0]
    return tr.data.astype(float), float(tr.stats.delta)

def load_channel(path: Path):
    if obspy_read is not None:
        try:
            return _load_obspy(path)
        except Exception:
            pass
    arr = np.loadtxt(path)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        t = arr[:, 0].astype(float)
        acc = arr[:, 1].astype(float)
        dt = float(np.median(np.diff(t)))
        return acc, dt
    raise RuntimeError(f"No pude leer {path}: instala obspy o usa TXT con columnas t,acc.")

def preprocess(acc: np.ndarray, dt: float, fmin: float = 0.1, fmax: float = 25.0) -> np.ndarray:
    acc = detrend(acc.astype(float), type='linear')
    fs = 1.0 / float(dt)
    acc = _bandpass(acc, fs, fmin=fmin, fmax=fmax)
    acc = np.nan_to_num(acc, copy=False)
    return acc

def normalize_units(acc_raw: np.ndarray):
    pga_raw = float(np.nanmax(np.abs(acc_raw)))
    if pga_raw > 5e3:          # µm/s^2
        return acc_raw * 1e-6, "µm/s^2", 1e-6
    elif 50 < pga_raw <= 5e3:  # gal = cm/s^2
        return acc_raw * 0.01, "cm/s^2", 0.01
    else:
        return acc_raw, "m/s^2", 1.0

def main():
    candE = list(DATA_DIR.glob("ESCTL.HNE.*"))
    candN = list(DATA_DIR.glob("ESCTL.HNN.*"))
    if len(candE) == 0 or len(candN) == 0:
        raise SystemExit(f"No encuentro ESCTL.HNE.* o ESCTL.HNN.* en {DATA_DIR}")

    pE = candE[0]; pN = candN[0]
    accE, dtE = load_channel(pE)
    accN, dtN = load_channel(pN)

    if abs(dtE - dtN) > 1e-12:
        dt = min(dtE, dtN)
        t_end = min(len(accE) * dtE, len(accN) * dtN)
        t = np.arange(0.0, t_end, dt)
        tE = np.arange(0.0, len(accE) * dtE, dtE)[:len(accE)]
        tN = np.arange(0.0, len(accN) * dtN, dtN)[:len(accN)]
        accE = np.interp(t, tE, accE).astype(float)
        accN = np.interp(t, tN, accN).astype(float)
    else:
        dt = dtE
        n = min(len(accE), len(accN))
        accE = accE[:n].astype(float)
        accN = accN[:n].astype(float)

    accE = preprocess(accE, dt)
    accN = preprocess(accN, dt)

    pgaE = float(np.max(np.abs(accE)))
    pgaN = float(np.max(np.abs(accN)))
    acc_raw = accE if pgaE >= pgaN else accN

    acc, unidad, factor = normalize_units(acc_raw)

    np.save(OUT_DIR / "acc.npy", acc)
    (OUT_DIR / "figs").mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "dt.txt", "w") as f:
        f.write(str(dt))

    pga = float(np.max(np.abs(acc)))
    print(f"[INFO] Canal usado: {'E' if pgaE>=pgaN else 'N'}; PGA_raw={max(pgaE,pgaN):.3g} ({unidad}), factor={factor}")
    print(f"[OK] outputs/acc.npy (N={len(acc)}), dt={dt:.6f} s, PGA={pga:.4f} m/s^2")
    if pga > 50:
        print("[ALERTA] PGA > 50 m/s^2: revisa unidades.")

if __name__ == "__main__":
    main()