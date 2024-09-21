import numpy as np

def _custom_mask():

    mask = np.ones((156, 156))
    mask_value = 0
    istep = -0.65
    jstep = -1
    istart = 25
    jstart = 85
    step = 7
    for c in range(0, 40):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = mask_value

    mask_value = 0
    istep = -0.65
    jstep = -1
    istart = 42
    jstart = 37
    step = 8
    for c in range(0, 40):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = mask_value

    mask_value = 0
    istep = -0.65
    jstep = -1
    istart = 86
    jstart = 15
    step = 8
    for c in range(0, 16):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = mask_value

    mask_value = 0
    istep = 0.65
    jstep = 1
    istart = 112
    jstart = 56
    step = 8
    for c in range(0, 50):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = mask_value

    mask_value = 0
    istep = -0.65
    jstep = -1
    istart = 113
    jstart = 144
    step = 7
    for c in range(0, 55):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = mask_value

    mask_value = 0
    istep = -0.65
    jstep = -1
    istart = 64
    jstart = 148
    step = 7
    for c in range(0, 25):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = mask_value

    mask_value = 0
    istep = 0.9
    jstep = 0.2
    istart = 60
    jstart = 110
    step = 8
    for c in range(0, 60):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = mask_value

    mask_value = 0
    istep = 0.9
    jstep = -0.7
    istart = 114
    jstart = 27
    step = 8
    for c in range(0, 37):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = mask_value

    return mask

_custom_mask_2m1134 = _custom_mask()
