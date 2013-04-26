import scipy
import matplotlib.pyplot as plt
import numpy as np

def tv_denoise(im, weight=50, eps=2.e-4, n_iter_max=200):
    """Perform total-variation denoising on 2D images.

    Parameters
    ----------
    im : ndarray
        Input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`)
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : ndarray
        Denoised array of floats.

    Notes
    -----
    The principle of total variation denoising is explained in
    http://en.wikipedia.org/wiki/Total_variation_denoising.

    This code is an implementation of the algorithm of Rudin, Fatemi and Osher
    that was proposed by Chambolle in [1]_.

    References
    ----------
    .. [1] A. Chambolle, An algorithm for total variation minimization and
           applications, Journal of Mathematical Imaging and Vision,
           Springer, 2004, 20, 89-97.

    Examples
    --------
    >>> from skimage import color, data
    >>> lena = color.rgb2gray(data.lena())
    >>> lena += 0.5 * lena.std() * np.random.randn(*lena.shape)
    >>> denoised_lena = denoise_tv(lena, weight=60)

    """

    px = np.zeros_like(im)
    py = np.zeros_like(im)
    gx = np.zeros_like(im)
    gy = np.zeros_like(im)
    d = np.zeros_like(im)
    i = 0
    while i < n_iter_max:
        d = -px - py
        d[1:] += px[:-1]
        d[:, 1:] += py[:, :-1]

        out = im + d
        E = (d**2).sum()
        gx[:-1] = np.diff(out, axis=0)
        gy[:, :-1] = np.diff(out, axis=1)
        norm = np.sqrt(gx**2 + gy**2)
        E += weight * norm.sum()
        norm *= 0.5 / weight
        norm += 1
        px -= 0.25 * gx
        px /= norm
        py -= 0.25 * gy
        py /= norm
        E /= float(im.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if np.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    print type(out)
    return out
    
    
def main():
    l = scipy.misc.lena()
    l = l[230:290, 220:320]

    noisy = l + 0.4*l.std()*np.random.random(l.shape)
    print type(noisy)
    tv_denoised = tv_denoise(noisy, weight=10)

    plt.figure(figsize=(12, 2.8))

    plt.subplot(131)
    plt.imshow(noisy, cmap=plt.cm.gray, vmin=40, vmax=220)
    plt.axis('off')
    plt.title('noisy', fontsize=20)
    plt.subplot(132)
    plt.imshow(tv_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
    plt.axis('off')
    plt.title('TV denoising', fontsize=20)

    tv_denoised = tv_denoise(noisy, weight=50)
    plt.subplot(133)
    plt.imshow(tv_denoised, cmap=plt.cm.gray, vmin=40, vmax=220)
    plt.axis('off')
    plt.title('(more) TV denoising', fontsize=20)

    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,
                        right=1)
    plt.show()
    
if __name__ == "__main__":
    main()
