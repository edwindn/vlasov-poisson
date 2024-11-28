import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, cumulative_trapezoid
from tqdm import tqdm
import pickle
import scipy
from scipy.interpolate import BSpline

Nx, Nv = 2000, 2000
k, alpha, vmax = 0.1, 0.5, 20.0  # Wavenumber, perturbation factor, cutoff velocity
dt = 0.025  # Time step
t_end = 100.0  # End time
L = 2 * np.pi / k  # Periodic structure length in X
sigma_v = 10

x = np.linspace(-L, L, Nx)
v = np.linspace(-vmax, vmax, Nv)
dx, dv = x[1] - x[0], v[1] - v[0]
N_steps = int(round(t_end / dt))
MIN_SHOW = 0.001

def extract_vals(f):
    rho = trapezoid(f, v, axis=1)
    g = -cumulative_trapezoid(rho, x, initial=0)
    g -= np.mean(g)
    phi = np.array([trapezoid(rho*abs(xval-x), x) for xval in x])
    Ekin = 0.5 * trapezoid(trapezoid(f*v**2, v, axis=1), x, axis=0)
    Epot = trapezoid(rho*phi, x)
    return rho, g, Ekin, Epot

def plot_phase_space(f, ref1, ref2, rho, cm1, cm2, title):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
    f = np.where(f > 5*MIN_SHOW, 2, 0) + np.where(ref1 > 5*MIN_SHOW, 5, 0) + np.where(ref2 > 5*MIN_SHOW, 5, 0)
    axs[0].pcolor(x, v, f[:-1, :-1].T, shading='auto', cmap='Reds')
    axs[0].set_title(title)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('v')
    axs[0].plot([cm1[0], cm2[0]], [cm1[1], cm2[1]], color='black', linestyle='-', linewidth=0.5)
    axs[1].fill_between(x, rho, color='blue')
    axs[1].set_title('rho')
    axs[1].legend()
    plt.savefig(f'{title}.png')
    plt.close()

def x_half_shift(f):
    for j in range(1, Nv-1):
        i = np.arange(1, Nx-1)
        sign = np.sign(v[j])
        shift = 0.5 * abs(v[j]) * dt/dx

        int_shift = int(np.floor(shift))
        shift -= int_shift

        f[:, j] = np.roll(f[:,j], int(sign * int_shift))
        # note circular shift is typical of periodic BCs

        Dxf = np.zeros(Nx)
        Dxf[i] = shift * (f[i,j] + sign * (f[i+1, j] - f[i-1, j]) * 0.25 * (1 - shift))
        Dxf[0], Dxf[-1] = Dxf[-2], Dxf[1] # change this - we don't need periodic BCs

        f[i,j] = f[i,j] + Dxf[i - int(sign)] - Dxf[i]

    return f

def v_shift(f, G):
    for i in range(1, Nx-1):
        j = np.arange(1, Nv-1)
        sign = np.sign(G[i])
        shift = abs(G[i]) * dt/dv

        int_shift = int(np.floor(shift))
        shift -= int_shift

        if sign > 0:
            f[i, int_shift:] = f[i, :Nv-int_shift]
            f[i, :int_shift] = 0.0
        else:
            f[i, :Nv-int_shift] = f[i, int_shift:]
            f[i, Nv-int_shift:] = 0.0

        Dvf = np.zeros(Nv)
        Dvf[j] = shift * (f[i,j] + sign * (f[i, j+1] - f[i, j-1]) * 0.25 * (1 - shift))
        Dvf[0], Dvf[-1] = 0.0, 0.0

        f[i,j] = f[i,j] + Dvf[j - int(sign)] - Dvf[j]

    return f

def advance(f, ref1, ref2):
    f, ref1, ref2 = x_half_shift(f), x_half_shift(ref1), x_half_shift(ref2)
    rho = trapezoid(f+ref1+ref2, v, axis=1)
    g = -cumulative_trapezoid(rho, x, initial=0)
    g -= np.mean(g)
    f, ref1, ref2 = v_shift(f, g), v_shift(ref1, g), v_shift(ref2, g)
    f, ref1, ref2 = x_half_shift(f), x_half_shift(ref1), x_half_shift(ref2)
    return f, ref1, ref2, rho


def initial_f(x, v):
    def func(x, v):
        return np.exp(-v**2 / 2*sigma_v**2) / np.sqrt(2 * np.pi) * (0.5 + alpha * np.cos(k * x))
    cond = (np.abs(x) < np.pi/k) & (np.abs(v) < 1/sigma_v)
    return np.where(cond, func(x, v), 0)

# ---

MIN_SHOW = 0.01

def find_cm(f):
    m = np.sum(f)
    xcm = np.sum(f * x[:, None]) / m
    vcm = np.sum(f * v[None, :]) / m
    return xcm, vcm

def find_inclination(ref1, ref2):
    xcm1, vcm1 = find_cm(ref1)
    xcm2, vcm2 = find_cm(ref2)
    return (vcm2 - vcm1) / (xcm2 - xcm1)

def plot_velocity(history):
    theta = -np.arctan(history)
    t = np.arange(len(history))

    shift = 0
    for i in range(1, len(theta)):
        if np.sign(theta[i]-shift) != np.sign(theta[i-1]-shift):
            theta[i:] += np.pi
            shift += np.pi

    theta_spline = BSpline(t, theta, 3)
    omega_spline = theta_spline.derivative()
    omega = omega_spline(t)
    # remove extreme values caused by discontinuities in theta
    omega = omega[omega < 0.1]
    t = np.arange(len(omega))
    plt.figure(figsize=(10,6))
    plt.plot(t, omega, color='red')
    plt.title("coiling velocity")
    plt.ylabel('radians / iteration')
    plt.savefig('angular.png')
    plt.close()

def get_times(history):
    theta = -np.arctan(history)
    times = []
    for i in range(1, len(theta)):
        if np.sign(theta[i]) != np.sign(theta[i-1]):
            times.append(i)
    times = np.diff(times)
    times = times[::2]
    t = np.arange(len(times))
    plt.figure(figsize=(10,6))
    plt.scatter(t, times, color='red')
    plt.title("turnaround time")
    plt.ylabel('number of iterations')
    plt.savefig('times.png')
    plt.close()

N_steps = 4000

def main():
    # plot inner turnaround time for different initial widths as function of evolution step
    with open('turnaround.pkl', 'rb') as file:
        history, last_state = pickle.load(file)

    plot_velocity(history)
    get_times(history)
    return

#def main():
    f = initial_f(x[:, None], v[None, :])
    refs = np.copy(f)
    width = 150
    xmid = int(f.shape[0]//2)
    vmid = int(f.shape[1]//2)
    xtop, xbottom = xmid + width//2, xmid - width//2

    f[xtop-4:xtop+4, vmid-2:vmid+2] = 0
    f[xbottom-4:xbottom+4, vmid-2:vmid+2] = 0
    refs = refs - f
    ref_right, ref_left = np.copy(refs), np.copy(refs)
    ref_right[:xmid, :] = 0
    ref_left[xmid:, :] = 0
    history = []

    ###
    #with open('turnaround.pkl', 'rb') as file:
    #    history, (f, ref_left, ref_right) = pickle.load(file)
    with tqdm(total=N_steps, desc="Progress") as global_bar:
        for T in range(N_steps):
            f, ref_left, ref_right, rho = advance(f, ref_left, ref_right)
            slope = find_inclination(ref_left, ref_right)
            history.append(slope)
            if T % 100 == 0:
                cm1, cm2 = find_cm(ref_left), find_cm(ref_right)
                plot_phase_space(f, ref_left, ref_right, rho, cm1, cm2, f't={T}, inclination={round(slope, 2)}')
            global_bar.update(1)

    with open('turnaround.pkl', 'wb') as file:
        pickle.dump((history, (f, ref_left, ref_right)), file)


if __name__ == '__main__':
    main()
