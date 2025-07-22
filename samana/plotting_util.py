import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['xtick.major.width'] = 3.5
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.minor.size'] = 6
plt.rcParams['ytick.minor.size'] = 6
plt.rcParams['ytick.major.width'] = 3.5
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20

def mock_lens_data_plot(image_sim, window_size, vminmax=1.5, cmap='gist_heat', label='', zd='', zs='',
                        save_fig=False, filename=None, x_image=None, y_image=None, image_labels=None,
                        delta_x_label=None, delta_y_label=None):
    fig = plt.figure(1)
    fig.set_size_inches(8, 8)
    ax = plt.subplot(111)
    extent = [-window_size / 2, window_size / 2, -window_size / 2, window_size / 2]
    image = deepcopy(image_sim)
    image[np.where(image < 10 ** -vminmax)] = 10 ** -vminmax
    im = ax.imshow(np.log10(image), origin='lower', alpha=0.75,
                   vmin=-vminmax, vmax=vminmax, cmap=cmap, extent=extent)
    ax.set_xticks([])
    ax.set_yticks([])
    xlow = -window_size / 2 + 0.2
    xhigh = xlow + 1.0
    y = -window_size / 2 + 0.3
    ax.plot([xlow, xhigh], [y, y], color='w', lw=4)
    ax.annotate('1 arcsec', xy=(xlow + 0.165, y - 0.21), fontsize=18, color='w')
    ax.annotate(label + '\n' + r'$z_{\rm{d}} = $' + str(zd) + '\n' + r'$z_{\rm{s}} = $' + str(zs), xy=(0.035, 0.78),
                xycoords='axes fraction', fontsize=24, color='w')
    cbar = plt.colorbar(im, fraction=0.046, pad=0.01)
    cbar.set_label(r'$\log_{10} \ \rm{flux}$', fontsize=25)
    if image_labels is None:
        image_labels = ['A', 'B', 'C', 'D']
    if delta_x_label is None:
        delta_x_label = [-0.08] *4
    if delta_y_label is None:
        delta_y_label = [0.1] * 4
    if x_image is not None:
        for i in range(0, 4):
            ax.annotate(image_labels[i], xy=(x_image[i]+delta_x_label[i], y_image[i]+delta_y_label[i]), color='w', fontsize=22)
    plt.tight_layout()
    if save_fig:
        plt.savefig(filename)
    plt.show()

def mock_substructure_plot(x_image, y_image, window_size, n_pixels, lens_model, lens_model_macro, kwargs_lens,
                           kwargs_lens_macro, label='', save_fig=False, filename=None, include_cbar=True):

    _r = np.linspace(-window_size / 2, window_size / 2, n_pixels)
    _xx, _yy = np.meshgrid(_r, _r)
    shape0 = _xx.shape
    kappa_macro = lens_model_macro.kappa(_xx.ravel(), _yy.ravel(), kwargs_lens_macro).reshape(shape0)
    kappa = lens_model.kappa(_xx.ravel(), _yy.ravel(), kwargs_lens).reshape(shape0)
    residual_kappa = kappa - kappa_macro
    mean = np.mean(residual_kappa[np.where(np.hypot(_xx, _yy) > 0.2)])

    fig = plt.figure(1)
    fig.set_size_inches(8, 8)
    ax = plt.subplot(111)
    extent = [-window_size / 2, window_size / 2, -window_size / 2, window_size / 2]

    im = ax.imshow(residual_kappa - mean, origin='lower', vmin=-0.1, vmax=0.1, cmap='seismic', alpha=1.,
                   extent=extent)
    ax.scatter(x_image, y_image, color='k', marker='x', s=200, alpha=0.8)
    ax.scatter(x_image, y_image, color='k', marker='+', s=250, alpha=0.8)

    ext = LensModelExtensions(lens_model)
    ra_crit_list, dec_crit_list, _, _ = ext.critical_curve_caustics(kwargs_lens, compute_window=window_size,
                                                                    grid_scale=0.01)
    for (racrit, deccrit) in zip(ra_crit_list[0:1], dec_crit_list[0:1]):
        ax.plot(racrit, deccrit, color='g', linestyle='--', lw=4)

    ax.set_xticks([])
    ax.set_yticks([])
    xlow = -window_size / 2 + 0.2
    xhigh = xlow + 1.0
    y = -window_size / 2 + 0.3
    ax.plot([xlow, xhigh], [y, y], color='k', lw=4)
    ax.annotate('1 arcsec', xy=(xlow + 0.165, y - 0.21), fontsize=18, color='k')
    if label is not None:
        ax.annotate(label, xy=(0.035, 0.9),
                    xycoords='axes fraction', fontsize=24, color='k',
                    bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.9, ec="k", lw=2))
    if include_cbar:
        cbar = plt.colorbar(im, fraction=0.046, pad=0.01, ticks=[-0.1, -0.05, 0.0, 0.05, 0.1])
        cbar.set_label(r'$\kappa - \kappa_{\rm{macro}}$', fontsize=25, labelpad=-2.5)
    # image_labels = ['A', 'B', 'C', 'D']
    # for i in range(0, 4):
    #     ax.annotate(image_labels[i], xy=(x_image[i]+0.05, y_image[i]+0.05), color='k', fontsize=15)

    plt.tight_layout()
    if save_fig:
        plt.savefig(filename)
        plt.show()
    return ax

