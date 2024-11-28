import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.interpolate import BSpline
from scipy.optimize import root_scalar
from tqdm import tqdm
import pickle
import scipy
from scipy.signal import find_peaks

"""
Gaussian has a characteristic length -> use power law distribution

!! enforce mass and probability conservation, and non-negativity

"""

Nx, Nv = 2000, 2000
k, alpha, vmax = 0.1, 0.5, 20.0  # Wavenumber, perturbation factor, cutoff velocity
dt = 0.025  # Time step
t_end = 100.0  # End time
L = 2 * np.pi / k
sigma_v = 50

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

def plot_single(f, rho, title):
    # Trim x and v values to only show |x| < x0 and |v| < v0
    x_trim_mask = np.abs(x) < 20
    rho_mask = np.abs(x) < 20
    v_trim_mask = np.abs(v) < 10
    x_trim = x[x_trim_mask]
    v_trim = v[v_trim_mask]
    f_trim = f[np.ix_(x_trim_mask, v_trim_mask)]

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=400)
    f_trim = np.where(f_trim > MIN_SHOW, f_trim, 0)
    pcm = axs[0].pcolor(x_trim, v_trim, f_trim[:-1, :-1].T, shading='auto', cmap='binary')
    pcm.set_clim(np.mean(f_trim), 0.05)
    axs[0].set_title(title)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('v')

    # Plot rho using the full x array (but you could trim it similarly if needed)
    axs[1].fill_between(x[rho_mask], rho[rho_mask], color='black')
    axs[1].set_title('rho')
    axs[1].legend()

    plt.savefig(f'{title}.png')
    plt.close()

def plot_two(f1, f2, rho1, rho2, title, rescale=False):
    # Trim x and v values to only show |x| < 20 and |v| < 20
    x_trim_mask = np.abs(x) < 20
    v_trim_mask = np.abs(v) < 20
    x_trim = x[x_trim_mask]
    v_trim = v[v_trim_mask]    
    f1_trim = f1[np.ix_(x_trim_mask, v_trim_mask)]
    f2_trim = f2[np.ix_(x_trim_mask, v_trim_mask)]
    f1_trim = np.where(f1_trim > MIN_SHOW, 1, 0)
    f2_trim = np.where(f2_trim > MIN_SHOW, 1.2, 0)

    if rescale:
        f2_trim = f2_trim[::2, ::2] # scale factor 0.5
        padx = (f1_trim.shape[0] - f2_trim.shape[0]) // 2
        padv = (f1_trim.shape[1] - f2_trim.shape[1]) // 2
        f2_trim = np.pad(f2_trim, ((padx, padx), (padv, padv)), mode='constant')
        if f2_trim.shape[0] < f1_trim.shape[0]:
            f2_trim = np.pad(f2_trim, ((0, 1), (0, 0)), mode='constant')
        elif f2_trim.shape[0] > f1_trim.shape[0]:
            f2_trim = f2_trim[:-1,:]
        if f2_trim.shape[1] < f1_trim.shape[1]:
            f2_trim = np.pad(f2_trim, ((0, 0), (0, 1)), mode='constant')
        elif f2_trim.shape[1] > f1_trim.shape[1]:
            f2_trim = f2_trim[:,:-1]

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=400)
    pcm1 = axs[0].pcolor(x_trim, v_trim, f1_trim[:-1, :-1].T, shading='auto', cmap='Reds', alpha=0.5)
    pcm2 = axs[0].pcolor(x_trim, v_trim, f2_trim[:-1, :-1].T, shading='auto', cmap='Reds', alpha=0.5)

    axs[0].set_title(title)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('v')

    if rescale:
        rho2 = rho2[::2]
        pad = (len(rho1) - len(rho2)) // 2
        rho2 = np.pad(rho2, (pad, pad), mode='constant')
        if len(rho2) < len(rho1):
            rho2 = np.pad(rho2, (0, 1), mode='constant')
        elif len(rho2) > len(rho1):
            rho2 = rho2[:-1]

    axs[1].fill_between(x_trim, rho1[x_trim_mask], color='blue', alpha=0.5, label='rho1')
    axs[1].fill_between(x_trim, rho2[x_trim_mask], color='red', alpha=0.5, label='rho2')
    axs[1].set_title('rho')
    axs[1].legend()

    plt.savefig(f'{title}.png')
    plt.close()

def plot_two(f1, f2, rho1, rho2, title, scale_factor):
    # Trim x and v values to only show |x| < 20 and |v| < 20
    x_trim_mask = np.abs(x) < 20
    v_trim_mask = np.abs(v) < 20
    x_trim = x[x_trim_mask]
    v_trim = v[v_trim_mask]    
    f1_trim = f1[np.ix_(x_trim_mask, v_trim_mask)]
    f2_trim = f2[np.ix_(x_trim_mask, v_trim_mask)]
    
    # Apply thresholding to make the functions uniform in color
    f1_trim = np.where(f1_trim > MIN_SHOW, 1, 0)
    f2_trim = np.where(f2_trim > MIN_SHOW, 1.2, 0)

    # Rescale f2_trim if requested
    if scale_factor != 1.0:
        # Rescale f2 using scipy.ndimage.zoom to allow arbitrary scale factors
        f2_trim = scipy.ndimage.zoom(f2_trim, scale_factor, order=1)  # Linear interpolation
        
        # Pad or trim f2_trim to match the size of f1_trim
        padx = (f1_trim.shape[0] - f2_trim.shape[0]) // 2
        padv = (f1_trim.shape[1] - f2_trim.shape[1]) // 2
        f2_trim = np.pad(f2_trim, ((padx, padx), (padv, padv)), mode='constant')

        if f2_trim.shape[0] < f1_trim.shape[0]:
            f2_trim = np.pad(f2_trim, ((0, 1), (0, 0)), mode='constant')
        elif f2_trim.shape[0] > f1_trim.shape[0]:
            f2_trim = f2_trim[:-1, :]
        
        if f2_trim.shape[1] < f1_trim.shape[1]:
            f2_trim = np.pad(f2_trim, ((0, 0), (0, 1)), mode='constant')
        elif f2_trim.shape[1] > f1_trim.shape[1]:
            f2_trim = f2_trim[:, :-1]

    fig, axs = plt.subplots(1, 2, figsize=(15, 5), dpi=400)

    # Plot f1 and f2 with transparency
    pcm1 = axs[0].pcolor(x_trim, v_trim, f1_trim[:-1, :-1].T, shading='auto', cmap='Blues', alpha=0.5)
    pcm2 = axs[0].pcolor(x_trim, v_trim, f2_trim[:-1, :-1].T, shading='auto', cmap='Reds', alpha=0.5)
    axs[0].text(0.5*x_trim.max(), 0.8*v_trim.max(), f'scale factor = {scale_factor}', fontsize=8, color='black')

    axs[0].set_title(title)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('v')

    # Rescale and align rho2 if requested
    if scale_factor != 1.0:
        rho2 = scipy.ndimage.zoom(rho2, scale_factor, order=1)
        
        pad = (len(rho1) - len(rho2)) // 2
        rho2 = np.pad(rho2, (pad, pad), mode='constant')

        if len(rho2) < len(rho1):
            rho2 = np.pad(rho2, (0, 1), mode='constant')
        elif len(rho2) > len(rho1):
            rho2 = rho2[:-1]

    axs[1].fill_between(x_trim, rho1[x_trim_mask], color='blue', alpha=0.5, label='rho1')
    axs[1].fill_between(x_trim, rho2[x_trim_mask], color='red', alpha=0.5, label='rho2')
    axs[1].set_title('rho')
    axs[1].legend()

    plt.savefig(f'{title}.png')
    plt.close()

def plot_triple(f1, f2, f3, title): # must be in right order
    x_trim_mask = np.abs(x) < 20
    v_trim_mask = np.abs(v) < 10
    x_trim = x[x_trim_mask]
    v_trim = v[v_trim_mask]
    f1_trim = f1[np.ix_(x_trim_mask, v_trim_mask)]
    f2_trim = f2[np.ix_(x_trim_mask, v_trim_mask)]
    f3_trim = f3[np.ix_(x_trim_mask, v_trim_mask)]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=400)
    f1_trim = np.where(f1_trim > MIN_SHOW, 1, 0)
    f2_trim = np.where(f2_trim > MIN_SHOW, 1, 0)
    f3_trim = np.where(f3_trim > MIN_SHOW, 1, 0)
    
    axs[0].pcolor(x_trim, v_trim, f1_trim[:-1, :-1].T, shading='auto', cmap='Reds')
    axs[1].pcolor(x_trim, v_trim, f2_trim[:-1, :-1].T, shading='auto', cmap='Reds')
    axs[2].pcolor(x_trim, v_trim, f3_trim[:-1, :-1].T, shading='auto', cmap='Reds')
    axs[0].set_title('Inner only')
    axs[1].set_title('Uniform mean field')
    axs[2].set_title('Full system')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('v')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('v')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('v')

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

def advance_two(f1, f2):
    f1, f2 = x_half_shift(f1), x_half_shift(f2)
    rho = trapezoid(f1+f2, v, axis=1)
    g = -cumulative_trapezoid(rho, x, initial=0)
    g -= np.mean(g)
    f1, f2 = v_shift(f1, g), v_shift(f2, g)
    f1, f2 = x_half_shift(f1), x_half_shift(f2)
    rho1, rho2 = trapezoid(f1, v, axis=1), trapezoid(f2, v, axis=1)
    return f1, f2, rho1, rho2

def advance(f):
    f = x_half_shift(f)
    rho = trapezoid(f, v, axis=1)
    g = -cumulative_trapezoid(rho, x, initial=0)
    g -= np.mean(g)
    f = v_shift(f, g)
    f = x_half_shift(f)
    return f, rho

def count_xcross(f, title='test', show=False, threshold=2e-4):
    mask = np.abs(x) < 40
    f = f[mask, :]
    f = np.maximum(f, 0)
    slice_width = 4
    f[:,Nx//2+slice_width:] = 0
    f[:,:Nv//2-slice_width] = 0
    rho = trapezoid(f, v, axis=1)
    peaks, _ = find_peaks(rho)
    xcross = []
    for p in peaks:
        if rho[p] > threshold: # lower the threshold for higher t due to numerical diffusion
            xcross.append((int(p)))
    return (len(xcross)-1)/2
    plt.figure(figsize=(10, 6))
    plt.scatter(x[mask], rho, label='Data', color='blue')
    for xc in xcross:
        plt.axvline(x=x[mask][xc], color='red', linestyle='--', linewidth=0.8, label='Peak' if xc == xcross[0] else "")  # Label only the first line
    plt.show()
    #plt.pause(1.3)
    plt.close()
    #plt.savefig(f'{title}.png', dpi=500)

    
def initial_f(x, v):
    def func(x, v):
        return np.exp(-v**2 / 2*sigma_v**2) / np.sqrt(2 * np.pi) * (0.5 - 0.01 * alpha * x**2)
    cond = (np.abs(x) < 10) & (np.abs(v) < 1 / sigma_v)
    return np.where(cond, func(x, v), 0)

def energy(f):
    rho = trapezoid(f, v, axis=1)
    g = -cumulative_trapezoid(rho, x, initial=0)
    g -= np.mean(g)
    phi = -cumulative_trapezoid(g, x, initial=0)
    phi -= np.min(phi)

    e_x1 = trapezoid(f*0.5*v**2, v, axis=1)
    e1 = trapezoid(e_x1, x)

    e_v1 = trapezoid(f.T*phi, x, axis=1)
    e2 = trapezoid(e_v1, v)
    e = e1 + e2
    return e
    plot_single(f, rho, 'test-rho')
    plot_single(f, g, 'test-g')
    plot_single(f, phi, 'test-phi')

# ---

MIN_SHOW = 0.01
N_steps = 5000

def main():
    #history = pickle.load(open('history1.pkl', 'rb')) + pickle.load(open('history2.pkl', 'rb')) + pickle.load(open('history3.pkl', 'rb'))
    history = pickle.load(open('long_time/history_fine.pkl', 'rb')) + pickle.load(open('long_time/history_fine2.pkl', 'rb')) + pickle.load(open('long_time/history_fine3.pkl', 'rb')) + pickle.load(open('long_time/history_fine4.pkl', 'rb'))
    print(len(history))
    crossings = []

    for f in history[:100]:
        c = count_xcross(f)
        crossings.append(c)
    for f in history[100:]:
        c = count_xcross(f, threshold=5e-5)
        crossings.append(c)
    plt.scatter(np.arange(len(crossings)), np.array(crossings))
    plt.title('All x')
    plt.show()
    plt.close()

    crossings = []
    for f in history:
        width = 200
        f[Nx//2-width:Nx//2+width, :] = 0
        c = count_xcross(f)
        crossings.append(c)
    plt.scatter(np.arange(len(crossings)), np.array(crossings))
    plt.title(f'|x| > {width}')
    plt.show()
    plt.close()
    return

    # evolve forward and repeat plotting
    # -> minimise resolution limitations, plot only for |x| > xmin

#def main():
    #f = initial_f(x[:, None], v[None, :])
    f = pickle.load(open('long_time/history_fine3.pkl', 'rb'))[-1]
    tot_p = np.sum(f)
    history = []
    

    with tqdm(total=N_steps, desc="Progress") as global_bar:
        for T in range(1, N_steps):
            f, rho = advance(f)
            #f = f / np.sum(f) * tot_p
            # how to enforce energy conservation ?

            if T % 100 == 0:
                e = energy(f)
                print(f'Energy: {e}')
                print(f'Density: {np.sum(f)}')
                history.append(np.copy(f))
                #plot_single(f, rho, f't={T+10700}')

            global_bar.update(1)
    
    with open('long_time/history_fine4.pkl', 'wb') as f:
        pickle.dump(history, f)


if __name__ == '__main__':
    main()
