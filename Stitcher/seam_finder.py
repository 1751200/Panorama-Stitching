import cv2
import numpy as np


def cal_energy_map(left_img, right_img):
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    sx = np.array([[-2, 0, 2], [-1, 0, 1], [-2, 0, 2]])
    sy = np.array([[-2, -1, -2], [0, 0, 0], [2, 1, 2]])

    left_img_sx = cv2.filter2D(left_img, -1, sx)
    left_img_sy = cv2.filter2D(left_img, -1, sy)
    right_img_sx = cv2.filter2D(right_img, -1, sx)
    right_img_sy = cv2.filter2D(right_img, -1, sy)

    e_color = (left_img - right_img) ** 2
    e_geometry = (left_img_sx - right_img_sx) * (left_img_sy - right_img_sy)
    e_map = e_color + e_geometry
    return e_map.astype(float)


def find_seam(left_img, right_img, min_indy, max_indy, min_indx, max_indx, is_debug=False):
    e_map = cal_energy_map(left_img, right_img)
    # init path weight with each e value of the first line in overlap area
    paths_weight = e_map[min_indy, min_indx:max_indx + 1].reshape(1, -1)
    # init path point index with
    paths = np.arange(min_indx, max_indx + 1).reshape(1, -1)  # save index
    for i in range(min_indy, max_indy + 1):
        # boundary process
        lefts_index = paths[-1, :] - 1
        lefts_index[lefts_index < 0] = 0
        rights_index = paths[-1, :] + 1
        rights_index[rights_index > max_indx - 1] = max_indx - 1
        mids_index = paths[-1, :]
        mids_index[mids_index < min_indx] = min_indx
        mids_index[mids_index > max_indx - 1] = max_indx - 1
        # compute next row strength value(remove begin and end point)
        lefts = e_map[i, lefts_index] + paths_weight[-1, :]
        mids = e_map[i, paths[-1, :]] + paths_weight[-1, :]
        rights = e_map[i, rights_index] + paths_weight[-1, :]
        # return the index of min strength value
        values_3direct = np.vstack((lefts, mids, rights))
        index_args = np.argmin(values_3direct, axis=0) - 1  #
        # next min strength value and index
        weights = np.min(values_3direct, axis=0)
        path_row = paths[-1, :] + index_args
        paths_weight = np.insert(paths_weight, paths_weight.shape[0], values=weights, axis=0)
        paths = np.insert(paths, paths.shape[0], values=path_row, axis=0)

    # search min path
    min_index = np.argmin(paths_weight[-1, :])
    best_seam = paths[:, min_index]
    if is_debug:
        for i in range(0, len(best_seam)):
            e_map[min_indy + i][best_seam[i]] = 255
        print(best_seam)
        cv2.imshow("e_map", e_map.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return best_seam
