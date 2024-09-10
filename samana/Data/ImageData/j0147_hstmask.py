import numpy as np

def _custom_mask():

    image_data_shape = (160, 160)
    mask = np.ones(image_data_shape)

    istep = -2.2
    jstep = 2
    istart = 89
    jstart = 85
    step = 12
    for c in range(0, 31):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = -1.8
    jstep = -2
    istart = 89
    jstart = 85
    step = 12
    for c in range(0, 41):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = -1.8
    jstep = -2
    istart = 101
    jstart = 60
    step = 8
    for c in range(0, 41):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = -1.8
    jstep = -2
    istart = 93
    jstart = 34
    step = 6
    for c in range(0, 19):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = 2
    jstep = -2.
    istart = 113
    jstart = 33
    step = 8
    for c in range(0, 19):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0
    istep = 2
    jstep = -1.8
    istart = 125
    jstart = 58
    step = 8
    for c in range(0, 14):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0
    istep = 2
    jstep = 2.0
    istart = 126
    jstart = 75
    step = 8
    for c in range(0, 14):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0
    istep = 2
    jstep = 2.0
    istart = 125
    jstart = 88
    step = 8
    for c in range(0, 14):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = 2
    jstep = 2.0
    istart = 125
    jstart = 123
    step = 8
    for c in range(0, 14):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = -2.1
    jstep = 2.0
    istart = 95
    jstart = 113
    step = 9
    for c in range(0, 20):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = 2
    jstep = -1.7
    istart = 120
    jstart = 92
    step = 8
    for c in range(0, 17):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0

    istep = -2.2
    jstep = 2.0
    istart = 51
    jstart = 87
    step = 8
    for c in range(0, 22):
        maskpixes_vert = np.arange(int(istart + istep * c), int(istart + step + istep * c))
        maskpixels_hor = np.arange(int(jstart + jstep * c), int(jstart + step + jstep * c))
        xx, yy = np.meshgrid(maskpixes_vert, maskpixels_hor)
        xx, yy = xx.ravel(), yy.ravel()
        mask[xx, yy] = 0.0
    return mask

_custom_mask_0147 = _custom_mask()
