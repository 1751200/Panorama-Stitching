"""
Multiple image stitching sample
===========================
Show results of image stitching using either advanced `OpenCV Stitcher` API or some low-level APIs along
with self-implemented features (e.g., Laplacian Pyramid Blending)
"""
from PIL import Image
from typing import Dict
import streamlit as st
import numpy as np
import time
import cv2

import doc
import Stitcher
import OCStitcher
import OpenCVStitcher

FILE_TYPES = ["png", "jpg", "jpeg"]
is_stitching = False


@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}


def cv2_read_img(io_bytes):
    io_bytes.seek(0)
    file_bytes = np.asarray(bytearray(io_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def main():
    """Run this function to run the app"""
    static_store = get_static_store()

    # st.sidebar.info("cv2 version: " + cv2.__version__)
    st.title("Multiple Image Stitching")
    st.sidebar.info(__doc__)
    st.sidebar.header('Navigation')
    nav = st.sidebar.radio(
        "Go to",
        ('Self-implemented Stitcher', 'OpenCV Stitcher API', 'OpenCV Advanced APIs', 'Doc')
    )

    if nav == 'Doc':
        st.sidebar.header('Document Content')
        doc_nav = st.sidebar.radio(
            "Section",
            ('Best Seam', 'Blending', 'Illumination Compensation')
        )
        if doc_nav == 'Best Seam':
            st.markdown(doc.best_seam)
            graph_cut = Image.open("doc/img/graphcut.png")
            st.image(graph_cut, use_column_width=True, caption='Graph Cut')
        elif doc_nav == 'Blending':
            st.markdown(doc.direct_blending)
            normal = Image.open("doc/img/normal.png")
            st.image(normal, use_column_width=True, caption='Direct Blending')
            st.markdown(doc.middle_line_blending)
            middle = Image.open("doc/img/middle.png")
            st.image(middle, use_column_width=True, caption='Middle Line Blinding')
            st.markdown(doc.weighted_average_blending)
            average = Image.open("doc/img/average.png")
            st.image(average, use_column_width=True, caption='Middle Line Blinding')
            st.markdown(doc.laplacian_blending1)
            pyramid = Image.open("doc/img/拉普拉斯金字塔.png")
            st.image(pyramid, use_column_width=True, caption='Laplacian Pyramid')
            st.markdown(doc.laplacian_blending2)
            laplacian = Image.open('doc/img/laplacian.png')
            st.image(laplacian, use_column_width=True, caption='Laplacian Blending')
            st.markdown(doc.laplacian_blending3)
            laplacian_graph_cut = Image.open("doc/img/laplacianWithGraphCut.png")
            st.image(laplacian_graph_cut, use_column_width=True, caption='Laplacian Blending with Graph Cut')
        else:
            st.markdown(doc.illumination_compensation)
            retinex = Image.open('doc/img/retinex.png')
            st.image(retinex, caption='Retinex')
            st.markdown(doc.illumination_compensation2)
    else:
        result = st.file_uploader("Upload your images", type=FILE_TYPES)
        if result:
            value = result.getvalue()
            # And add it to the static_store if not already in
            if value not in static_store.values():
                static_store[result] = value
        else:
            # Hack to clear list if the user clears the cache and reloads the page
            static_store.clear()

        if st.button("Clear image list"):
            static_store.clear()
        if st.checkbox("Show image list?", True):
            st.write(list(static_store.keys()))
        if st.checkbox("Show content of images?"):
            images = []
            for key in static_store.keys():
                images.append(cv2_read_img(key))
            if images:
                st.image(images, width=int(630/len(images)), channels='BGR',
                         caption=[str(i) for i in range(len(images))])

        status = st.empty()
        if len(static_store) < 2:
            status.info("Upload one or more images of type ." + ", .".join(FILE_TYPES))
        else:
            # global is_stitching
            # is_stitching = True
            status.warning("Stitching images...Wait a few seconds...")
            start_time = time.time()
            if nav == 'Self-implemented Stitcher':
                st.sidebar.header("Stitching Parameters")
                show_matching_result = st.sidebar.checkbox("Show matching results")
                show_stitching_progresses = st.sidebar.checkbox('Show stitching progresses')
                show_masks = st.sidebar.checkbox('Show masks (only available for laplacian blender with graph cut)')
                feature = st.sidebar.selectbox(
                    "Select a type of features used for images matching",
                    ('SIFT', 'SURF', 'ORB', 'BRISK', 'AKAZE')
                )
                blender = st.sidebar.selectbox(
                    "Select a blend method",
                    ('laplacian', 'laplacianWithGraphCut','average', 'middle', 'normal')
                )
                light_compensation = st.sidebar.selectbox(
                    "Select a light compensation method",
                    ('grey', 'hist', 'MSRCR', 'AWB')
                )
                s = Stitcher.Stitch([cv2_read_img(key) for key in static_store.keys()])
                s.leftshift(feature=feature, blender=blender, light=light_compensation)
                s.rightshift(feature=feature, blender=blender, light=light_compensation)
                matches = []
                progresses = []
                masks = []
                for image in s.matches:
                    matches.append(image)
                if matches and show_matching_result:
                    st.header("Feature Matching")
                    st.image(matches, use_column_width=True, channels='BGR',
                             caption=[str(i) for i in range(len(matches))])
                for image in s.progresses:
                    progresses.append(image)
                if progresses and show_stitching_progresses:
                    st.header("Stitching progresses")
                    st.image(progresses, use_column_width=True, channels='BGR',
                             caption=[str(i) for i in range(len(progresses))])
                for image in s.masks:
                    masks.append(image)
                if masks and show_masks:
                    st.header("Masks")
                    st.image(masks, use_column_width=True, channels='BGR')
                st.header("Stitching Result")
                st.image(s.leftImage, use_column_width=True, channels='BGR', caption='Stitch Result')
            elif nav == 'OpenCV Stitcher API':
                st.sidebar.header("Stitching parameters")
                show_source_code = st.sidebar.checkbox("Show source code")
                stitched = OpenCVStitcher.stitch([cv2_read_img(key) for key in static_store.keys()])
                st.image(stitched, use_column_width=True, channels='BGR', caption='Stitch Result')
                if show_source_code:
                    st.code(OpenCVStitcher.code)
            elif nav == 'OpenCV Advanced APIs':
                st.sidebar.header("Stitching Parameters")
                feature = st.sidebar.selectbox(
                    "Select a type of features used for images matching",
                    ('SIFT', 'SURF', 'ORB', 'BRISK', 'AKAZE')
                )
                matcher = st.sidebar.selectbox(
                    "Select a matcher used for pairwise image matching",
                    ('homography', 'affine')
                )
                if feature == 'ORB':
                    match_conf = st.sidebar.slider("Confidence for feature matching step", 0.1, 0.9, 0.3)
                else:
                    match_conf = st.sidebar.slider("Confidence for feature matching step", 0.1, 0.9, 0.65)
                warp_type = st.sidebar.selectbox(
                    "Select a warp surface type",
                    OCStitcher.WARP_CHOICES
                )
                seam = st.sidebar.selectbox(
                    "Select a seam estimation method",
                    list(OCStitcher.SEAM_FIND_CHOICES.keys())
                )
                blend_type = st.sidebar.selectbox(
                    "Select a blend method",
                    ('multiband', 'feature')
                )
                blend_strength = st.sidebar.slider(
                    "Blending strength", 0, 100, 5
                )
                s = OCStitcher.Stitch([cv2_read_img(key) for key in static_store.keys()])
                s.features = feature.lower()
                s.matcher_type = matcher
                s.match_conf = match_conf
                s.warp = warp_type
                s.seam = seam
                s.blend = blend_type
                s.blend_strength = blend_strength
                st.image(s.stitch(), use_column_width=True, channels='BGR', caption='Stitch Result')
            else:
                pass
            end_time = time.time()
            status.success("Stitching completed successfully in " +
                           str(round(end_time - start_time, 2)) + "s")


main()
