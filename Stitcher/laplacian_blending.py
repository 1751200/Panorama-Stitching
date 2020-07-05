import cv2
import numpy as np


def laplacian_blending(img1, img2, mask, levels=4):
    G1 = np.float32(img1)
    G2 = np.float32(img2)
    GM = np.float32(mask)
    gaussPyr1 = [G1]
    gaussPyr2 = [G2]
    gaussPyrM = [GM]

    # Generate Gaussian Pyramids
    for i in range(levels):
        G1 = cv2.pyrDown(G1)
        G2 = cv2.pyrDown(G2)
        GM = cv2.pyrDown(GM)
        gaussPyr1.append(G1)
        gaussPyr2.append(G2)
        gaussPyrM.append(GM)

    # Generate Laplacian Pyramids
    laplacianPyr1 = [gaussPyr1[levels - 1]]
    laplacianPyr2 = [gaussPyr2[levels - 1]]
    laplacianPyrM = [gaussPyrM[levels - 1]]
    for i in range(levels - 1, 0, -1):
        dstsize = (gaussPyr1[i-1].shape[1], gaussPyr1[i-1].shape[0])
        temp_pyrup1 = cv2.pyrUp(gaussPyr1[i], dstsize=dstsize)
        temp_pyrup2 = cv2.pyrUp(gaussPyr2[i], dstsize=dstsize)
        L1 = np.subtract(gaussPyr1[i - 1], temp_pyrup1)
        L2 = np.subtract(gaussPyr2[i - 1], temp_pyrup2)
        laplacianPyr1.append(L1)
        laplacianPyr2.append(L2)
        laplacianPyrM.append(gaussPyrM[i - 1])

    # Now blend images according to mask in each level
    LS = []
    for l1, l2, lm in zip(laplacianPyr1, laplacianPyr2, laplacianPyrM):
        ls = l1 * lm + l2 * (1.0 - lm)
        LS.append(ls)

    # Now reconstruct
    ls_reconstruct = LS[0]
    for i in range(1, levels):
        ls_reconstruct = cv2.pyrUp(ls_reconstruct, dstsize=(LS[i].shape[1], LS[i].shape[0]))
        ls_reconstruct = cv2.add(ls_reconstruct, LS[i])

    return np.uint8(np.clip(ls_reconstruct, 0, 255))
