"""Simple manual Hodgkin-Huxley implementation.

This module implements the HH equations with hand-written ODE integration
(RK4 by default). It intentionally does not use the project's optimized
integrators so you can see the ODE steps explicitly.
"""
from math import exp
import numpy as np


class HHManual:
    """Manual Hodgkin-Huxley solver.

    Methods:
        run(T, dt, stimulus, method='rk4') -> dict of arrays
    """

    def __init__(self,
            C_m=1.0,
            g_Na=120.0,
            g_K=36.0,
            g_L=0.3,
            E_Na=50.0,
            E_K=-77.0,
            E_L=-54.387,
            V0=-65.0):
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.V0 = V0

    # Rate functions matching hh_core.models
    def alpha_m(self, V):
        x = V + 40.0
        if abs(x) < 1e-8:
            return 1.0
        return 0.1 * x / (1.0 - exp(-x / 10.0))

    def beta_m(self, V):
        return 4.0 * exp(-(V + 65.0) / 18.0)

    def alpha_h(self, V):
        return 0.07 * exp(-(V + 65.0) / 20.0)

    def beta_h(self, V):
        return 1.0 / (1.0 + exp(-(V + 35.0) / 10.0))

    def alpha_n(self, V):
        x = V + 55.0
        if abs(x) < 1e-8:
            return 0.1
        return 0.01 * x / (1.0 - exp(-x / 10.0))

    def beta_n(self, V):
        return 0.125 * exp(-(V + 65.0) / 80.0)

    def steady_gates(self, V):
        am = self.alpha_m(V)
        bm = self.beta_m(V)
        ah = self.alpha_h(V)
        bh = self.beta_h(V)
        an = self.alpha_n(V)
        bn = self.beta_n(V)
        m = am / (am + bm)
        h = ah / (ah + bh)
        n = an / (an + bn)
        return m, h, n

    def ionic_currents(self, V, m, h, n):
        I_Na = self.g_Na * (m ** 3) * h * (V - self.E_Na)
        I_K = self.g_K * (n ** 4) * (V - self.E_K)
        I_L = self.g_L * (V - self.E_L)
        I_ion = I_Na + I_K + I_L
        return I_Na, I_K, I_L, I_ion

    def derivatives(self, V, m, h, n, I_ext):
        # dV/dt, dm/dt, dh/dt, dn/dt
        I_Na, I_K, I_L, I_ion = self.ionic_currents(V, m, h, n)
        dVdt = (I_ext - I_ion) / self.C_m

        am = self.alpha_m(V)
        bm = self.beta_m(V)
        ah = self.alpha_h(V)
        bh = self.beta_h(V)
        an = self.alpha_n(V)
        bn = self.beta_n(V)

        dmdt = am * (1.0 - m) - bm * m
        dhdt = ah * (1.0 - h) - bh * h
        dndt = an * (1.0 - n) - bn * n

        return dVdt, dmdt, dhdt, dndt

    def rk4_step(self, V, m, h, n, I_ext, dt):
        # k1
        k1 = self.derivatives(V, m, h, n, I_ext)
        # k2
        V2 = V + 0.5 * dt * k1[0]
        m2 = m + 0.5 * dt * k1[1]
        h2 = h + 0.5 * dt * k1[2]
        n2 = n + 0.5 * dt * k1[3]
        k2 = self.derivatives(V2, m2, h2, n2, I_ext)
        # k3
        V3 = V + 0.5 * dt * k2[0]
        m3 = m + 0.5 * dt * k2[1]
        h3 = h + 0.5 * dt * k2[2]
        n3 = n + 0.5 * dt * k2[3]
        k3 = self.derivatives(V3, m3, h3, n3, I_ext)
        # k4
        V4 = V + dt * k3[0]
        m4 = m + dt * k3[1]
        h4 = h + dt * k3[2]
        n4 = n + dt * k3[3]
        k4 = self.derivatives(V4, m4, h4, n4, I_ext)

        V_new = V + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        m_new = m + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        h_new = h + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        n_new = n + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])

        return V_new, m_new, h_new, n_new

    def euler_step(self, V, m, h, n, I_ext, dt):
        dVdt, dmdt, dhdt, dndt = self.derivatives(V, m, h, n, I_ext)
        return V + dt * dVdt, m + dt * dmdt, h + dt * dhdt, n + dt * dndt

    def run(self, T, dt, stimulus, method='rk4'):
        """Run simulation.

        stimulus: array-like or callable(t)
        method: 'rk4' or 'euler'
        """
        # validate inputs
        if T <= 0:
            raise ValueError('T must be positive')
        if dt <= 0:
            raise ValueError('dt must be positive')
        if method not in ('rk4', 'euler'):
            raise ValueError("method must be 'rk4' or 'euler'")

        n_steps = int(np.ceil(T / dt))
        time = np.linspace(0.0, T, n_steps + 1)

        V = float(self.V0)
        m, h, n = self.steady_gates(V)

        V_rec = np.empty(n_steps + 1)
        m_rec = np.empty(n_steps + 1)
        h_rec = np.empty(n_steps + 1)
        n_rec = np.empty(n_steps + 1)
        I_rec = np.empty(n_steps + 1)

        V_rec[0] = V
        m_rec[0] = m
        h_rec[0] = h
        n_rec[0] = n

        # prepare stimulus array
        if callable(stimulus):
            I_arr = np.array([float(stimulus(t)) for t in time[:-1]], dtype=float)
        else:
            s = np.asarray(stimulus, dtype=float)
            if s.ndim == 1:
                # truncate or pad to n_steps
                if len(s) < n_steps:
                    I_arr = np.pad(s, (0, n_steps - len(s)), mode='edge')
                else:
                    I_arr = s[:n_steps]
            else:
                # assume shape (n_steps, batch) or (n_steps,)
                if s.shape[0] < n_steps:
                    # pad rows
                    pad_rows = n_steps - s.shape[0]
                    I_arr = np.pad(s, ((0, pad_rows), (0, 0)), mode='edge')[:, 0]
                else:
                    I_arr = s[:n_steps, 0]

        for i in range(n_steps):
            I_ext = float(I_arr[i]) if i < len(I_arr) else 0.0

            if method == 'rk4':
                V, m, h, n = self.rk4_step(V, m, h, n, I_ext, dt)
            else:
                V, m, h, n = self.euler_step(V, m, h, n, I_ext, dt)

            V_rec[i + 1] = V
            m_rec[i + 1] = m
            h_rec[i + 1] = h
            n_rec[i + 1] = n
            I_rec[i] = I_ext

        # final I
        I_rec[-1] = I_rec[-2] if len(I_rec) > 1 else 0.0

        return {
            'time': time,
            'V': V_rec,
            'm': m_rec,
            'h': h_rec,
            'n': n_rec,
            'I': I_rec,
        }
