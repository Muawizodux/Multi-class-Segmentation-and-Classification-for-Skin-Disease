import numpy as np

labels = {0: [108, 0, 115],
          1: [145, 1, 122],
          2: [216, 47, 148],
          3: [254, 246, 242],
          4: [181, 9, 130],
          5: [236, 85, 157],
          6: [73, 0, 106],
          7: [248, 123, 168],
          8: [0, 0, 0],
          9: [127, 255, 255],
          10: [127, 255, 142],
          11: [255, 127, 127]}


def encoded_mask(mask: np.ndarray):
    row, col = mask.shape[0:2]
    r, g, b = mask[:, :, 0], mask[:, :, 1], mask[:, :, 2]
    single_mask = np.zeros_like(r)
    single_mask[single_mask == 0] = 8

    for i in range(row):
        for j in range(col):
            for k, pixel in labels.items():
                if (r[i][j] == pixel[0]) and (g[i][j] == pixel[1]) and (b[i][j] == pixel[2]):
                    single_mask[i][j] = k

    return single_mask


def decoded_mask(mask: np.ndarray):
    # mask should be mapped with np.argmax() already
    row, col = mask.shape[0:2]
    b = np.zeros_like(mask)
    g = np.zeros_like(mask)
    r = np.zeros_like(mask)
    b[b == 0] = 8
    g[g == 0] = 8
    r[r == 0] = 8
    for i in range(row):
        for j in range(col):
            for k, pixel in labels.items():
                if mask[i][j] == k:
                    b[i][j] = pixel[0]
                    g[i][j] = pixel[1]
                    r[i][j] = pixel[2]
    single_mask = np.zeros((row, col, 3), dtype=np.uint8)
    single_mask[:, :, 0] = b
    single_mask[:, :, 1] = g
    single_mask[:, :, 2] = r

    return single_mask

