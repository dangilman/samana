import numpy as np

def _custom_mask():

    image_data_shape = (58, 58)
    mask = np.ones(image_data_shape)

    istep = 0.0
    jstep = 1
    istart = 12
    jstart = 30
    step = 6
    for c in range(0, 20):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = 0.0
    jstep = 1
    istart = 21
    jstart = 23
    step = 6
    for c in range(0, 29):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = 0.0
    jstep = -1
    istart = 21
    jstart = 7
    step = 6
    for c in range(0, 8):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = 0.
    jstep = -1
    istart = 31
    jstart = 3
    step = 7
    for c in range(0, 4):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = -1.
    jstep = 0
    istart = 13
    jstart = 14
    step = 6
    for c in range(0, 13):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = 0.
    jstep = -1
    istart = 13
    jstart = 14
    step = 6
    for c in range(0, 14):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = -1.
    jstep = 0
    istart = 53
    jstart = 15
    step = 5
    for c in range(0, 14):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = -1.
    jstep = 0
    istart = 54
    jstart = 11
    step = 4
    for c in range(0, 9):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = -1.
    jstep = 0
    istart = 54
    jstart = 23
    step = 4
    for c in range(0, 21):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = 0.0
    jstep = 1
    istart = 33
    jstart = 19
    step = 4
    for c in range(0, 9):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = -1.0
    jstep = 0
    istart = 28
    jstart = 29
    step = 4
    for c in range(0, 6):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    return mask

_custom_mask_0147 = _custom_mask()

