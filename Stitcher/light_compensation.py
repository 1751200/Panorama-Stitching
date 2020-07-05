import numpy as np
import cv2 as cv
import random
import sys


# GreyWorld
def grey_world(nimg):
    nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
    avgB = np.average(nimg[0])
    avgG = np.average(nimg[1])
    avgR = np.average(nimg[2])
    avg = (avgB + avgG + avgR) / 3
    nimg[0] = np.minimum(nimg[0] * (avg / avgB), 255)
    nimg[1] = np.minimum(nimg[1] * (avg / avgG), 255)
    nimg[2] = np.minimum(nimg[2] * (avg / avgR), 255)

    return  nimg.transpose(1, 2, 0).astype(np.uint8)


# Histogram Equalization
def hist_equalization(img):
    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
    channels = cv.split(ycrcb)
    cv.equalizeHist(channels[0], channels[0])  # equalizeHist(in,out)
    cv.merge(channels, ycrcb)
    img_eq = cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR)

    return img_eq


# Retinex
# SSR
def single_scale_retinex(img, sigma):
    temp = cv.GaussianBlur(img, (0, 0), sigma)
    gaussian = np.where(temp == 0, 0.01, temp)
    retinex = np.log10(img + 0.01) - np.log10(gaussian)

    return retinex


# MSR
def multi_scale_retinex(img, sigma_list):
    retinex = np.zeros_like(img*1.0)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)
    retinex = retinex / len(sigma_list)

    return retinex


def color_restoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return restoration


def simplest_color_balance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img


# MSRCR
def MSRCR(img, sigma_list=None, G=5, b=25, alpha=125, beta=46, low_clip=0.01, high_clip=0.99):
    if sigma_list is None:
        sigma_list = [16, 128, 256]
    img = np.float64(img) + 1.0
    img_retinex = multi_scale_retinex(img, sigma_list)
    img_color = color_restoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)
    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * 255
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplest_color_balance(img_msrcr, low_clip, high_clip)
    return img_msrcr


# AWB
def white_balance(img):
    rows = img.shape[0]
    cols = img.shape[1]
    final = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    avgA = np.average(final[:, :, 1])
    avgB = np.average(final[:, :, 2])
    for x in range(final.shape[0]):
        for y in range(final.shape[1]):
            l, a, b = final[x, y, :]
            l *= 100 / 255.0
            final[x, y, 1] = a - ((avgA - 128) * (l / 100.0) * 1.1)
            final[x, y, 2] = b - ((avgB - 128) * (l / 100.0) * 1.1)
    final = cv.cvtColor(final, cv.COLOR_LAB2BGR)

    return final


def calc_saturation(diff,slope,limit):
    ret = diff * slope
    if ret > limit:
        ret = limit
    elif ret < (-limit):
        ret = -limit
    return ret


def automatic_color_equalization(nimg, slope=10, limit=1000, samples=500):
    nimg = nimg.transpose(2, 0, 1)
    nimg = np.ascontiguousarray(nimg, dtype=np.uint8)
    width = nimg.shape[2]
    height = nimg.shape[1]
    cary = []
    for i in range(0, samples):
        _x = random.randint(0, width) % width
        _y = random.randint(0, height) % height
        dict = {"x": _x, "y": _y}
        cary.append(dict)

    mat = np.zeros((3, height, width), float)
    r_max = sys.float_info.min
    r_min = sys.float_info.max
    g_max = sys.float_info.min
    g_min = sys.float_info.max
    b_max = sys.float_info.min
    b_min = sys.float_info.max

    for i in range(height):
        for j in range(width):
            r = nimg[0, i, j]
            g = nimg[1, i, j]
            b = nimg[2, i, j]
            r_rscore_sum = 0.0
            g_rscore_sum = 0.0
            b_rscore_sum = 0.0
            denominator = 0.0

            for _dict in cary:
                _x = _dict["x"]  # width
                _y = _dict["y"]  # height
                dist = np.sqrt(np.square(_x - j) + np.square(_y - i))
                if (dist < height / 5):
                    continue
                _sr = nimg[0, _y, _x]
                _sg = nimg[1, _y, _x]
                _sb = nimg[2, _y, _x]
                r_rscore_sum += calc_saturation(int(r) - int(_sr), slope, limit) / dist
                g_rscore_sum += calc_saturation(int(g) - int(_sg), slope, limit) / dist
                b_rscore_sum += calc_saturation(int(b) - int(_sb), slope, limit) / dist
                denominator += limit / dist

            r_rscore_sum = r_rscore_sum / denominator
            g_rscore_sum = g_rscore_sum / denominator
            b_rscore_sum = b_rscore_sum / denominator
            mat[0, i, j] = r_rscore_sum
            mat[1, i, j] = g_rscore_sum
            mat[2, i, j] = b_rscore_sum
            if r_max < r_rscore_sum:
                r_max = r_rscore_sum
                if r_min > r_rscore_sum:
                    r_min = r_rscore_sum
                if g_max < g_rscore_sum:
                    g_max = g_rscore_sum
                if g_min > g_rscore_sum:
                    g_min = g_rscore_sum
                if b_max < b_rscore_sum:
                    b_max = b_rscore_sum
                if b_min > b_rscore_sum:
                    b_min = b_rscore_sum
    for i in range(height):
        for j in range(width):
            nimg[0, i, j] = (mat[0, i, j] - r_min) * 255 / (r_max - r_min)
            nimg[1, i, j] = (mat[1, i, j] - g_min) * 255 / (g_max - g_min)
            nimg[2, i, j] = (mat[2, i, j] - b_min) * 255 / (b_max - b_min)

    return nimg.transpose(1, 2, 0).astype(np.uint8)