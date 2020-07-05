from collections import OrderedDict
import cv2 as cv
import numpy as np

FEATURES_FIND_CHOICES = OrderedDict()
try:
    FEATURES_FIND_CHOICES['surf'] = cv.xfeatures2d_SURF.create
except AttributeError:
    print("SURF not available")
# if SURF not available, ORB is default
FEATURES_FIND_CHOICES['orb'] = cv.ORB.create
try:
    FEATURES_FIND_CHOICES['sift'] = cv.xfeatures2d_SIFT.create
except AttributeError:
    print("SIFT not available")
try:
    FEATURES_FIND_CHOICES['brisk'] = cv.BRISK_create
except AttributeError:
    print("BRISK not available")
try:
    FEATURES_FIND_CHOICES['akaze'] = cv.AKAZE_create
except AttributeError:
    print("AKAZE not available")

ESTIMATOR_CHOICES = OrderedDict()
ESTIMATOR_CHOICES['homography'] = cv.detail_HomographyBasedEstimator
ESTIMATOR_CHOICES['affine'] = cv.detail_AffineBasedEstimator

BA_COST_CHOICES = OrderedDict()
BA_COST_CHOICES['ray'] = cv.detail_BundleAdjusterRay
BA_COST_CHOICES['reproj'] = cv.detail_BundleAdjusterReproj
BA_COST_CHOICES['affine'] = cv.detail_BundleAdjusterAffinePartial
BA_COST_CHOICES['no'] = cv.detail_NoBundleAdjuster

WAVE_CORRECT_CHOICES = ('horiz', 'no', 'vert',)

WARP_CHOICES = (
    'spherical',
    'plane',
    'affine',
    'cylindrical',
    'fisheye',
    'stereographic',
    'compressedPlaneA2B1',
    'compressedPlaneA1.5B1',
    'compressedPlanePortraitA2B1',
    'compressedPlanePortraitA1.5B1',
    'paniniA2B1',
    'paniniA1.5B1',
    'paniniPortraitA2B1',
    'paniniPortraitA1.5B1',
    'mercator',
    'transverseMercator',
)

SEAM_FIND_CHOICES = OrderedDict()
SEAM_FIND_CHOICES['gc_color'] = cv.detail_GraphCutSeamFinder('COST_COLOR')
SEAM_FIND_CHOICES['gc_colorgrad'] = cv.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
SEAM_FIND_CHOICES['dp_color'] = cv.detail_DpSeamFinder('COLOR')
SEAM_FIND_CHOICES['dp_colorgrad'] = cv.detail_DpSeamFinder('COLOR_GRAD')
SEAM_FIND_CHOICES['voronoi'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM)
SEAM_FIND_CHOICES['no'] = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)

EXPOS_COMP_CHOICES = OrderedDict()
EXPOS_COMP_CHOICES['gain_blocks'] = cv.detail.ExposureCompensator_GAIN_BLOCKS
EXPOS_COMP_CHOICES['gain'] = cv.detail.ExposureCompensator_GAIN
EXPOS_COMP_CHOICES['channel'] = cv.detail.ExposureCompensator_CHANNELS
EXPOS_COMP_CHOICES['channel_blocks'] = cv.detail.ExposureCompensator_CHANNELS_BLOCKS
EXPOS_COMP_CHOICES['no'] = cv.detail.ExposureCompensator_NO

BLEND_CHOICES = ('multiband', 'feather', 'no',)


class Stitch:
    def __init__(self, images):
        self.images = images
        self.try_cuda = False
        self.work_megapix = 0.6
        self.features = list(FEATURES_FIND_CHOICES.keys())[1]
        self.matcher_type = 'homography'
        self.estimator = list(ESTIMATOR_CHOICES.keys())[0]
        self.match_conf = 0.3
        self.conf_thresh = 1.0

        self.ba = list(BA_COST_CHOICES.keys())[0]

        self.wave_correct = WAVE_CORRECT_CHOICES[0]

        self.warp = WARP_CHOICES[0]

        self.seam_megapix = 0.1
        self.seam = list(SEAM_FIND_CHOICES.keys())[0]
        self.compose_megapix = -1

        self.expos_comp = list(EXPOS_COMP_CHOICES.keys())[0]
        self.expos_comp_nr_feeds = 1
        self.expos_comp_nr_filtering = 2
        self.expos_comp_block_size = 32

        self.blend = BLEND_CHOICES[0]
        self.blend_strength = 5

        self.output = 'result.jpg'

    def get_matcher(self):
        if self.matcher_type == 'affine':
            matcher = cv.detail_AffineBestOf2NearestMatcher(False, self.try_cuda, self.match_conf)
        else:
            matcher = cv.detail.BestOf2NearestMatcher_create(self.try_cuda, self.match_conf)
        return matcher

    def get_compensator(self):
        expos_comp_type = EXPOS_COMP_CHOICES[self.expos_comp]
        if expos_comp_type == cv.detail.ExposureCompensator_CHANNELS:
            compensator = cv.detail_ChannelsCompensator(self.expos_comp_nr_feeds)
        elif expos_comp_type == cv.detail.ExposureCompensator_CHANNELS_BLOCKS:
            compensator = cv.detail_BlocksChannelsCompensator(
                self.expos_comp_block_size, self.expos_comp_block_size,
                self.expos_comp_nr_feeds
            )
        else:
            compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
        return compensator

    def show_result(self, result):
        zoom_x = 600.0 / result.shape[1]
        dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        dst = cv.resize(dst, dsize=None, fx=zoom_x , fy=zoom_x)
        cv.imshow(self.output, dst)
        cv.waitKey()

    def stitch(self):
        if self.wave_correct == 'no':
            do_wave_correct = False
        else:
            do_wave_correct = True

        images = self.images[:]
        full_img_sizes = []
        finder = FEATURES_FIND_CHOICES[self.features]()
        seam_work_aspect = 1
        features = []
        is_work_scale_set = False
        is_seam_scale_set = False
        is_compose_scale_set = False
        for idx, full_img in enumerate(images):
            full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
            if self.work_megapix < 0:
                img = full_img
                work_scale = 1
                is_work_scale_set = True
            else:
                if is_work_scale_set is False:
                    work_scale = min(1.0, np.sqrt(self.work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                    is_work_scale_set = True
                img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
            if is_seam_scale_set is False:
                seam_scale = min(1.0, np.sqrt(self.seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                seam_work_aspect = seam_scale / work_scale
                is_seam_scale_set = True
            img_feat =cv.detail.computeImageFeatures2(finder, img)
            features.append(img_feat)
            img = cv.resize(src=img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
            images[idx] = img

        matcher = self.get_matcher()
        p = matcher.apply2(features)
        matcher.collectGarbage()

        indices = cv.detail.leaveBiggestComponent(features, p, 0.3)
        img_subset = []
        full_img_sizes_subset = []
        for i in range(len(indices)):
            img_subset.append(images[indices[i, 0]])
            full_img_sizes_subset.append(full_img_sizes[indices[i, 0]])
        images = img_subset
        full_img_sizes = full_img_sizes_subset
        num_images = len(images)
        if num_images < 2:
            print("Need more images!")
            exit()

        estimator = ESTIMATOR_CHOICES[self.estimator]()
        b, cameras = estimator.apply(features, p, None)
        if not b:
            print("Homography estimation failed.")
            exit()
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)

        adjuster = BA_COST_CHOICES[self.ba]()
        adjuster.setConfThresh(1)
        refine_mask = np.zeros((3, 3), np.uint8)
        refine_mask[0, :] = 1
        refine_mask[1, 0:2] = 1
        adjuster.setRefinementMask(refine_mask)
        b, cameras = adjuster.apply(features, p, cameras)
        if not b:
            print("Camera parameters adjusting failed.")
            exit()
        focals = []
        for cam in cameras:
            focals.append(cam.focal)
        sorted(focals)
        if len(focals) % 2 == 1:
            warped_image_scale = focals[len(focals) // 2]
        else:
            warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
        if do_wave_correct:
            rmats = []
            for cam in cameras:
                rmats.append(np.copy(cam.R))
            rmats = cv.detail.waveCorrect(rmats, cv.detail.WAVE_CORRECT_HORIZ)
            for idx, cam in enumerate(cameras):
                cam.R = rmats[idx]

        corners = []
        masks_warped = []
        images_warped = []
        sizes = []
        masks = []
        for i in range(0, num_images):
            um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
            masks.append(um)

        warper = cv.PyRotationWarper(self.warp, warped_image_scale * seam_work_aspect)
        for idx in range(0, num_images):
            K = cameras[idx].K().astype(np.float32)
            swa = seam_work_aspect
            K[0, 0] *= swa
            K[0, 2] *= swa
            K[1, 1] *= swa
            K[1, 2] *= swa
            corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            corners.append(corner)
            sizes.append((image_wp.shape[1], image_wp.shape[0]))
            images_warped.append(image_wp)
            p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            masks_warped.append(mask_wp.get())

        images_warped_f = []
        for img in images_warped:
            imgf = img.astype(np.float32)
            images_warped_f.append(imgf)

        compensator = self.get_compensator()
        compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

        seam_finder = SEAM_FIND_CHOICES[self.seam]
        seam_finder.find(images_warped_f, corners, masks_warped)
        compose_scale = 1
        corners = []
        sizes = []
        blender = None
        for idx, img in enumerate(self.images):
            if not is_compose_scale_set:
                if self.compose_megapix > 0:
                    compose_scale = min(1.0, np.sqrt(self.compose_megapix * 1e6 / (img.shape[0] * img.shape[1])))
                is_compose_scale_set = True
                compose_work_aspect = compose_scale / work_scale
                warped_image_scale *= compose_work_aspect
                warper = cv.PyRotationWarper(self.warp, warped_image_scale)
                for i in range(0, len(images)):
                    cameras[i].focal *= compose_work_aspect
                    cameras[i].ppx *= compose_work_aspect
                    cameras[i].ppy *= compose_work_aspect
                    sz = (full_img_sizes[i][0] * compose_scale, full_img_sizes[i][1] * compose_scale)
                    K = cameras[i].K().astype(np.float32)
                    roi = warper.warpRoi(sz, K, cameras[i].R)
                    corners.append(roi[0:2])
                    sizes.append(roi[2:4])
            if abs(compose_scale - 1) > 1e-1:
                compose_img = cv.resize(src=img, dsize=None, fx=compose_scale, fy=compose_scale, interpolation=cv.INTER_LINEAR_EXACT)
            else:
                compose_img = img
            K = cameras[idx].K().astype(np.float32)
            corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
            mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
            p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
            compensator.apply(idx, corners[idx], image_warped, mask_warped)
            image_warped_s = image_warped.astype(np.int16)
            dilated_mask = cv.dilate(masks_warped[idx], None)
            seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
            mask_warped = cv.bitwise_and(seam_mask, mask_warped)
            if blender is None:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
                blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * self.blend_strength / 100
                if blend_width < 1:
                    blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
                elif self.blend == 'multiband':
                    blender = cv.detail_MultiBandBlender()
                    blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
                elif self.blend == "feather":
                    blender = cv.detail_FeatherBlender()
                    blender.setSharpness(1. / blend_width)
                blender.prepare(dst_sz)
            blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])

        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        # cv.imwrite(self.output, result)
        zoom_x = 600.0 / result.shape[1]
        dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        dst = cv.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)
        # cv.imshow(self.output, dst)
        # cv.waitKey()
        return dst


if __name__ == '__main__':
    img1 = cv.imread('input/1.jpg')
    img2 = cv.imread('input/2.jpg')
    img3 = cv.imread('input/3.jpg')
    images = []
    images.append(img1)
    images.append(img2)
    images.append(img3)
    stitch = Stitch(images)
    stitch.stitch()
