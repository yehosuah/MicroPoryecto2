# src/03_spectrum.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
CSV_DIR = OUT_DIR / "csv"; CSV_DIR.mkdir(parents=True, exist_ok=True)
FIGS = OUT_DIR / "figs"; FIGS.mkdir(parents=True, exist_ok=True)

def sdof_rk(acc_g: np.ndarray, dt: float, T: float, zeta: float = 0.05):
    acc_g = np.asarray(acc_g, dtype=float)
    dt = float(dt)
    w = 2.0 * np.pi / float(T)
    t = np.arange(len(acc_g)) * dt
    ag_fun = interp1d(t, acc_g, kind="linear", fill_value=(acc_g[0], acc_g[-1]), bounds_error=False)

    def f(ti, y):
        u, v = y
        ag = float(ag_fun(ti))
        a_rel = -2.0 * zeta * w * v - (w ** 2) * u - ag
        return [v, a_rel]

    sol = solve_ivp(f, (t[0], t[-1]), [0.0, 0.0], t_eval=t, method="RK45", rtol=1e-7, atol=1e-9)
    u = sol.y[0]; v = sol.y[1]
    ag = ag_fun(sol.t)
    a_rel = -2.0 * zeta * w * v - (w ** 2) * u - ag
    return u, v, a_rel

def response_spectrum(acc_g: np.ndarray, dt: float, periods: np.ndarray, zeta: float = 0.05):
    Sd = np.zeros_like(periods)
    Sv = np.zeros_like(periods)
    Sa = np.zeros_like(periods)
    for i, T in enumerate(periods):
        u, v, a = sdof_rk(acc_g, dt, T, zeta=zeta)
        Sd[i] = float(np.max(np.abs(u)))
        Sv[i] = float(np.max(np.abs(v)))
        Sa[i] = float(np.max(np.abs(a)))
    return Sd, Sv, Sa

def main():
    acc = np.load(OUT_DIR / "acc.npy")
    dt = float((OUT_DIR / "dt.txt").read_text())

    periods = np.logspace(np.log10(0.05), np.log10(4.0), 200)
    zeta = 0.05

    Sd, Sv, Sa = response_spectrum(acc, dt, periods, zeta=zeta)

    arr = np.column_stack([periods, Sd, Sv, Sa])
    np.savetxt(CSV_DIR / "spectra_5pct.csv", arr, delimiter=",", header="T,Sd,Sv,Sa", comments="")

    plt.figure(figsize=(8,5)); plt.loglog(periods, Sd); plt.xlabel("T (s)"); plt.ylabel("Sd (m)")
    plt.grid(True, which="both"); plt.tight_layout(); plt.savefig(FIGS / "Sd_vs_T.png", dpi=160)

    plt.figure(figsize=(8,5)); plt.loglog(periods, Sv); plt.xlabel("T (s)"); plt.ylabel("Sv (m/s)")
    plt.grid(True, which="both"); plt.tight_layout(); plt.savefig(FIGS / "Sv_vs_T.png", dpi=160)

    plt.figure(figsize=(8,5)); plt.loglog(periods, Sa); plt.xlabel("T (s)"); plt.ylabel("Sa (m/s²)")
    plt.grid(True, which="both"); plt.tight_layout(); plt.savefig(FIGS / "Sa_vs_T.png", dpi=160)

    print("[OK] Guardé CSV y figuras en outputs/")

if __name__ == "__main__":
    main()