# src/02_sdof_solver.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import argparse

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
FIGS = OUT_DIR / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

def sdof_rk(acc_g: np.ndarray, dt: float, T: float, zeta: float = 0.05, u0: float = 0.0, v0: float = 0.0):
    acc_g = np.asarray(acc_g, dtype=float)
    dt = float(dt)
    if T <= 0 or dt <= 0:
        raise ValueError("T y dt deben ser positivos.")
    w = 2.0 * np.pi / float(T)
    t = np.arange(len(acc_g)) * dt
    ag_fun = interp1d(t, acc_g, kind="linear", fill_value=(acc_g[0], acc_g[-1]), bounds_error=False)

    def f(ti, y):
        u, v = y
        ag = float(ag_fun(ti))
        a_rel = -2.0 * zeta * w * v - (w ** 2) * u - ag
        return [v, a_rel]

    sol = solve_ivp(f, (t[0], t[-1]), [u0, v0], t_eval=t, method="RK45", rtol=1e-7, atol=1e-9)
    u = sol.y[0]; v = sol.y[1]
    ag = ag_fun(sol.t)
    a_rel = -2.0 * zeta * w * v - (w ** 2) * u - ag
    a_abs = a_rel + ag
    return u, v, a_rel, a_abs, t, ag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=float, required=True)
    parser.add_argument("--zeta", type=float, default=0.05)
    args = parser.parse_args()

    acc = np.load(OUT_DIR / "acc.npy")
    dt = float((OUT_DIR / "dt.txt").read_text())

    u, v, a_rel, a_abs, t, ag = sdof_rk(acc, dt, args.T, zeta=args.zeta)

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(t, ag);      axs[0].set_ylabel("ag (m/s²)")
    axs[1].plot(t, u);       axs[1].set_ylabel("u (m)")
    axs[2].plot(t, v);       axs[2].set_ylabel("v (m/s)")
    axs[3].plot(t, a_rel);   axs[3].set_ylabel("a_rel (m/s²)"); axs[3].set_xlabel("t (s)")
    fig.suptitle(f"SDOF T={args.T:.3f}s, zeta={args.zeta*100:.1f}%")
    fig.tight_layout()
    out = FIGS / f"tiempo_T{args.T:.3f}.png"
    fig.savefig(out, dpi=160)
    print(f"[OK] Figura guardada: {out}")

if __name__ == "__main__":
    main()