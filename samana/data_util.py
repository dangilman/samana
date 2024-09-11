import numpy as np
import sys

def mask_quasar_images(baseline_mask, x_image, y_image, ra_grid, dec_grid, mask_size):

    if not isinstance(mask_size, list):
        mask_size = [mask_size] * len(x_image)
    for (xi, yi, r) in zip(x_image, y_image, mask_size):
        dx = abs(xi - ra_grid)
        dy = abs(yi - dec_grid)
        dr = np.hypot(dx, dy)
        inds_mask = np.where(dr <= r)
        baseline_mask[inds_mask] = 0.0
    return baseline_mask

def create_image_data_file(filename, image_data, psf_model, psf_error_map):
    np.set_printoptions(threshold=sys.maxsize)
    with open(filename, 'w') as f:
        f.write('import numpy as np\n')
        f.write('\n')
        f.write('image_data = np.'+str(repr(image_data)))
        if psf_model is not None:
            f.write('\n')
            f.write('psf_model = np.' + str(repr(psf_model)))
        if psf_error_map is not None:
            f.write('\n')
            f.write('psf_error_map = np.' + str(repr(psf_error_map)))
            f.write('\n')
    f.close()
