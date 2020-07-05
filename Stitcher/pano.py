import copy
import imutils
import numpy as np
import cv2
import sys
from .matchers import matchers
import time
from .laplacian_blending import laplacian_blending
from .seam_finder import find_seam


class Stitch:
    def __init__(self, images):
        # self.path = args
        # fp = open(self.path, 'r')
        # filenames = [each.rstrip('\r\n') for each in fp.readlines()]
        # print(filenames)
        # self.images = [cv2.resize(cv2.imread(each), (480, 320)) for each in filenames]
        self.images = images
        self.count = len(self.images)
        self.left_list, self.right_list, self.center_im = [], [], None
        self.matcher_obj = matchers()
        self.prepare_lists()

        self.matches = []
        self.progresses = []
        self.masks = []

    def prepare_lists(self):
        self.centerIdx = self.count / 2
        # self.center_im = self.images[int(self.centerIdx)]
        for i in range(self.count):
            if i <= self.centerIdx:
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])

    def leftshift(self, blender="average", feature="BRISK"):
        a = self.left_list[0]
        for b in self.left_list[1:]:
            H, matches = self.matcher_obj.match(a, b, feature, 'left')
            self.matches.append(matches)
            xh = np.linalg.inv(H)
            ds = np.dot(xh, np.array([
                [0, 0, a.shape[1], a.shape[1]],
                [0, a.shape[0], 0, a.shape[0]],
                [1, 1, 1, 1],
            ]))
            ds = ds[:][:] / ds[-1][:]
            maxX = max(ds[0][:])
            minX = min(ds[0][:])
            maxY = max(ds[1][:])
            minY = min(ds[1][:])

            trans = np.array([
                [1, 0, -minX],
                [0, 1, -minY],
                [0, 0, 1]
            ])

            dsize = (int(maxX - minX) + b.shape[0], int(maxY - minY) + b.shape[1])
            tmp = cv2.warpPerspective(a, trans.dot(xh), dsize)
            base = cv2.warpPerspective(b, trans, dsize)

            if blender == "average":
                tm, _ = self.averageBlender(tmp, base, dsize, "left")

            if blender == "laplacian":
                mask = np.ones_like(a, dtype=np.float32)
                mask = cv2.warpPerspective(mask, trans.dot(xh), dsize)
                tmp = self.laplacianBlender(tmp, base, mask, 4)

            if blender == "laplacianWithGraphCut":
                mask1 = np.logical_and(np.ones_like(tmp), tmp)
                mask2 = np.logical_and(np.ones_like(base), base)
                overlap = np.logical_and(mask1, mask2)
                overlap = np.float32(overlap)

                min_indy, max_indy, min_indx, max_indx = self.findCorner(overlap)
                path = find_seam(tmp, base, min_indy, max_indy, min_indx, max_indx)

                mask3 = np.float32(np.uint8(mask1) - overlap)
                overlap = np.float32(overlap)

                mask = self.getPathMask(path, overlap, min_indy, max_indy, min_indx, max_indx)

                self.masks.append(np.uint8(np.clip(mask, 0, 255)))
                mask += mask3

                tmp = self.laplacianBlender(tmp, base, mask, 4)

            if blender == "middle":
                tmp = self.middleBlender(tmp, base, dsize, "left")

            if blender == "normal":
                tmp = self.mix_and_match(base, tmp)

            tmp = self.culling(tmp)
            self.progresses.append(tmp)
            a = tmp

        self.leftImage = a

    def rightshift(self, blender="average", feature="BRISK"):
        for each in self.right_list:
            H, matches = self.matcher_obj.match(self.leftImage, each, feature, 'right')
            self.matches.append(matches)
            txyz = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            txyz = txyz / txyz[-1]
            dsize = (max(int(txyz[0]), self.leftImage.shape[1]), max(int(txyz[1]), self.leftImage.shape[0]))

            tmp = cv2.warpPerspective(each, H, dsize)
            base = np.zeros((tmp.shape[0], tmp.shape[1], 3), dtype=np.uint8)
            base[:self.leftImage.shape[0], :self.leftImage.shape[1], :] = self.leftImage

            if blender == "average":
                tmp, _ = self.averageBlender(tmp, base, dsize, "right")

            if blender == "laplacian":
                mask = np.ones_like(each, dtype=np.float32)
                mask = cv2.warpPerspective(mask, H, dsize)
                tmp = self.laplacianBlender(tmp, base, mask, 4)

            if blender == "laplacianWithGraphCut":
                mask1 = np.logical_and(np.ones_like(base), base)
                mask2 = np.logical_and(np.ones_like(tmp), tmp)
                overlap = np.logical_and(mask1, mask2)
                overlap = np.float32(overlap)

                min_indy, max_indy, min_indx, max_indx = self.findCorner(overlap)
                path = find_seam(tmp, base, min_indy, max_indy, min_indx, max_indx)
                mask3 = np.float32(np.uint8(mask1) - overlap)
                overlap = np.float32(overlap)

                mask = self.getPathMask(path, overlap, min_indy, max_indy, min_indx, max_indx)
                self.masks.append(np.uint8(np.clip(mask, 0, 255)))

                mask += mask3

                tmp = self.laplacianBlender(base, tmp, mask, 4)

            if blender == "middle":
                tmp = self.middleBlender(tmp, base, dsize, "right")

            if blender == "normal":
                tmp = self.mix_and_match(base, tmp)

            tmp = self.culling(tmp)
            self.progresses.append(tmp)
            self.leftImage = tmp

    def mix_and_match(self, leftImage, warpedImage):
        i1y, i1x = leftImage.shape[:2]
        i2y, i2x = warpedImage.shape[:2]

        t = time.time()
        black_l = np.where(leftImage == np.array([0, 0, 0]))
        black_wi = np.where(warpedImage == np.array([0, 0, 0]))

        for i in range(0, i1x):
            for j in range(0, i1y):
                try:
                    if np.array_equal(leftImage[j, i], np.array([0, 0, 0])) and np.array_equal(warpedImage[j, i],
                                                                                               np.array([0, 0, 0])):
                        # print "BLACK"
                        # instead of just putting it with black,
                        # take average of all nearby values and avg it.
                        warpedImage[j, i] = [0, 0, 0]
                    else:
                        if np.array_equal(warpedImage[j, i], [0, 0, 0]):
                            # print "PIXEL"
                            warpedImage[j, i] = leftImage[j, i]
                        else:
                            if not np.array_equal(leftImage[j, i], [0, 0, 0]):
                                bw, gw, rw = warpedImage[j, i]
                                bl, gl, rl = leftImage[j, i]
                                # bl = (bl+bw)/2
                                # gl = (gl+gw)/2
                                # rl = (rl+rw)/2
                                warpedImage[j, i] = [bl, gl, rl]
                except:
                    pass
        # cv2.imshow("waRPED mix", warpedImage)
        # cv2.waitKey()
        return warpedImage

    def trim_left(self):
        pass

    def findCorner(self, mask):
        for col in range(0, mask.shape[1]):
            if mask[:, col].any():
                min_x = col
                break
        for col in range(mask.shape[1] - 1, 0, -1):
            if mask[:, col].any():
                max_x = col
                break

        for row in range(0, mask.shape[0]):
            if mask[row, :].any():
                min_y = row
                break
        for row in range(mask.shape[0] - 1, 0, -1):
            if mask[row, :].any():
                max_y = row
                break

        return min_y, max_y, min_x, max_x

    def optimal_seam_rule_value(self, I1, I2):
        # I1 = np.int16(I1)
        # I2 = np.int16(I2)
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
        Sx = np.array([[-2, 0, 2], [-1, 0, 1], [-2, 0, 2]])
        Sy = np.array([[-2, -1, -2], [0, 0, 0], [2, 1, 2]])

        I1_Sx = cv2.filter2D(I1, -1, Sx)
        I1_Sy = cv2.filter2D(I1, -1, Sy)
        I2_Sx = cv2.filter2D(I2, -1, Sx)
        I2_Sy = cv2.filter2D(I2, -1, Sy)

        E_color = (I1 - I2) ** 2
        E_geometry = ((I1_Sx - I2_Sx)) * ((I1_Sy - I2_Sy))
        # E_geometry = 0
        E = E_color + E_geometry
        return E.astype(float)

    def optimal_seam_rule2(self, I1, I2):
        E = self.optimal_seam_rule_value(I1, I2)
        # optimal seam
        paths_weight = E[0, 1:-1].reshape(1, -1)  # Cumulative strength value
        paths = np.arange(1, E.shape[1] - 1).reshape(1, -1)  # save index
        for i in range(1, E.shape[0]):
            # boundary process
            lefts_index = paths[-1, :] - 1
            lefts_index[lefts_index < 0] = 0
            rights_index = paths[-1, :] + 1
            rights_index[rights_index > E.shape[1] - 1] = E.shape[1] - 1
            mids_index = paths[-1, :]
            mids_index[mids_index < 0] = 0
            mids_index[mids_index > E.shape[1] - 1] = E.shape[1] - 1

            # compute next row strength value(remove begin and end point)
            lefts = E[i, lefts_index] + paths_weight[-1, :]
            mids = E[i, paths[-1, :]] + paths_weight[-1, :]
            rights = E[i, rights_index] + paths_weight[-1, :]
            # return the index of min strength value
            values_3direct = np.vstack((lefts, mids, rights))
            index_args = np.argmin(values_3direct, axis=0) - 1  #
            # next min strength value and index
            weights = np.min(values_3direct, axis=0)
            path_row = paths[-1, :] + index_args
            paths_weight = np.insert(paths_weight, paths_weight.shape[0], values=weights, axis=0)
            paths = np.insert(paths, paths.shape[0], values=path_row, axis=0)
        # search min path
        paths_weight[np.where(paths_weight == 0)] = np.inf
        min_index = np.argmin(paths_weight[-1, :])
        return paths[:, min_index]

    def removal_seam(self, img_trans, img_targ, transform_corners, threshold=5):
        # img_trans warpPerspective image
        # img_targ target image
        # transform_corners the 4 corners of warpPerspective image

        # corners_orig = np.array([[0, 0, 1],
        #                         [0, img.shape[0], 1],
        #                         [img.shape[1], 0, 1],
        #                         [img.shape[1], img.shape[0], 1]])
        # obtain 4 corners from T transform

        pano = copy.deepcopy(img_trans)
        pano[0:img_targ.shape[0], 0:img_targ.shape[1]] = img_targ

        x_right = img_targ.shape[1]
        x_left = int(min(transform_corners[0, 0], transform_corners[0, 1]))
        rows = pano.shape[0]
        # calculate weight matrix
        alphas = np.array([x_right - np.arange(x_left, x_right)] * rows) / (x_right - x_left)
        alpha_matrix = np.ones((alphas.shape[0], alphas.shape[1], 3))
        alpha_matrix[:, :, 0] = alphas
        alpha_matrix[:, :, 1] = alphas
        alpha_matrix[:, :, 2] = alphas
        # common area one image no pixels
        alpha_matrix[img_trans[0:rows, x_left:x_right, :] <= threshold] = 1

        img_targ = pano[:, 0:img_targ.shape[1]]
        pano[0:rows, x_left:x_right] = img_targ[0:rows, x_left:x_right] * alpha_matrix \
                                       + img_trans[0:rows, x_left:x_right] * (1 - alpha_matrix)

        return pano

    def averageBlender(self, tmp, base, dsize, direction, mask=None):
        # Blending 1
        left = 0
        right = 0
        if mask is None:
            mask = np.zeros_like(tmp)
        if direction == "left":
            for col in range(0, dsize[0]):
                if base[:, col].any() and tmp[:, col].any():
                    left = col
                    break
            for col in range(dsize[0] - 1, 0, -1):
                if base[:, col].any() and tmp[:, col].any():
                    right = col
                    break

            for row in range(0, dsize[1]):
                for col in range(left, dsize[0]):
                    if not base[row, col].any():
                        tmp[row, col] = tmp[row, col]
                    elif not tmp[row, col].any():
                        tmp[row, col] = base[row, col]
                    else:
                        baseImgLen = float(abs(col - right))
                        tmpImgLen = float(abs(col - left))
                        alpha = baseImgLen / (baseImgLen + tmpImgLen)
                        mask[row, col] = alpha
                        tmp[row, col] = np.clip(base[row, col] * (1 - alpha) + tmp[row, col] * alpha, 0, 255)
        else:
            for col in range(0, dsize[0]):
                if base[:, col].any() and tmp[:, col].any():
                    left = col
                    break
            for col in range(dsize[0] - 1, 0, -1):
                if base[:, col].any() and tmp[:, col].any():
                    right = col
                    break

            for row in range(0, dsize[1]):
                for col in range(0, right):
                    if not base[row, col].any():
                        tmp[row, col] = tmp[row, col]
                    elif not tmp[row, col].any():
                        tmp[row, col] = base[row, col]
                    else:
                        baseImgLen = float(abs(col - left))
                        tmpImgLen = float(abs(col - right))
                        alpha = baseImgLen / (baseImgLen + tmpImgLen)
                        mask[row, col] = alpha
                        tmp[row, col] = np.clip(base[row, col] * (1 - alpha) + tmp[row, col] * alpha, 0, 255)

        return tmp, mask

    def laplacianBlender(self, tmp, base, mask, levels):
        tmp = laplacian_blending(tmp, base, mask, levels)
        return tmp

    def middleBlender(self, tmp, base, dsize, direction):
        # Blending 2
        left = 0
        right = 0
        if direction == "left":
            for col in range(0, dsize[0]):
                if base[:, col].any() and tmp[:, col].any():
                    left = col
                    break
            for col in range(dsize[0] - 1, 0, -1):
                if base[:, col].any() and tmp[:, col].any():
                    right = col
                    break

            for row in range(0, dsize[1]):
                for col in range(left, dsize[0]):
                    if not base[row, col].any():
                        tmp[row, col] = tmp[row, col]
                    elif not tmp[row, col].any():
                        tmp[row, col] = base[row, col]
                    else:
                        baseImgLen = float(abs(col - right))
                        tmpImgLen = float(abs(col - left))
                        alpha = baseImgLen / (baseImgLen + tmpImgLen)
                        if alpha < 0.5:
                            tmp[row, col] = base[row, col]
        else:
            left = 0
            right = 0
            for col in range(0, dsize[0]):
                if base[:, col].any() and tmp[:, col].any():
                    left = col
                    break
            for col in range(dsize[0] - 1, 0, -1):
                if base[:, col].any() and tmp[:, col].any():
                    right = col
                    break

            for row in range(0, dsize[1]):
                for col in range(0, right):
                    if not base[row, col].any():
                        tmp[row, col] = tmp[row, col]
                    elif not tmp[row, col].any():
                        tmp[row, col] = base[row, col]
                    else:
                        baseImgLen = float(abs(col - left))
                        tmpImgLen = float(abs(col - right))
                        alpha = baseImgLen / (baseImgLen + tmpImgLen)
                        if alpha < 0.5:
                            tmp[row, col] = base[row, col]

        return tmp

    def getPathMask(self, path, mask, min_indy, max_indy, min_indx, max_indx):
        for i in range(min_indy, max_indy + 1):
            for j in range(min_indx, max_indx + 1):
                if mask[i, j].any():
                    x = path[i - min_indy]
                    weight = 0.5
                    if j < x:
                        index = np.where(mask[i, :] == 1)[0][0]
                        weight = 0.5 + float(x - j) / float(x - index) * 0.5
                    else:
                        index = np.where(mask[i, :] == 1)[-1][0]
                        weight = 0.5 - float(j - x) / float(max_indx - x) * 0.5
                        if weight < 0:
                            weight = 0
                    mask[i, j] = weight

        return mask

    def culling(self, tmp):
        gray = cv2.cvtColor(tmp.copy(), cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]

        x1 = tmp.shape[1]
        x2 = 0
        y1 = tmp.shape[0]
        y2 = 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            x1 = x if x < x1 else x1
            y1 = y if y < y1 else y1
            x2 = (x + w) if (x + w) > x2 else x2
            y2 = (y + h) if (y + h) > y2 else y2
            # cv2.rectangle(tmp, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
        tmp = tmp[y1: y2, x1: x2]

        return tmp


if __name__ == '__main__':
    try:
        args = sys.argv[1]
    except:
        args = "txtlists/files3.txt"
    finally:
        print("Parameters : ", args)
    s = Stitch(args)
    s.leftshift(blender="laplacianWithGraphCut")
    s.rightshift(blender="laplacianWithGraphCut")
    print("done")
    cv2.imwrite("test35.jpg", s.leftImage)
    print("image written")
    cv2.destroyAllWindows()
