"""Streamlit v. 0.52 ships with a first version of a **file uploader** widget. You can find the
**documentation**
[here](https://streamlit.io/docs/api.html?highlight=file%20upload#streamlit.file_uploader).

For reference I've implemented an example of file upload here. It's available in the gallery at
[awesome-streamlit.org](https://awesome-streamlit.org).
"""
from typing import Dict
import streamlit as st
import numpy as np
import cv2

import Stitcher

FILE_TYPES = ["png", "jpg", "jpeg"]


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

    st.sidebar.title("Multiple Image Stitching")

    st.info(__doc__)
    result = st.file_uploader("Upload your images", type=FILE_TYPES)
    if result:
        # Process you file here
        value = result.getvalue()

        # And add it to the static_store if not already in
        if value not in static_store.values():
            static_store[result] = value
    else:
        static_store.clear()  # Hack to clear list if the user clears the cache and reloads the page
        st.info("Upload one or more images of type ." + ", .".join(FILE_TYPES))

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
    if st.button("Stitch Images"):
        s = Stitcher.Stitch([cv2_read_img(key) for key in static_store.keys()])
        s.leftshift()
        s.rightshift()
        st.image(s.leftImage, use_column_width=True, channels='BGR', caption='Stitch Result')

main()
