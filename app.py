"""Streamlit v. 0.52 ships with a first version of a **file uploader** widget. You can find the
**documentation**
[here](https://streamlit.io/docs/api.html?highlight=file%20upload#streamlit.file_uploader).

For reference I've implemented an example of file upload here. It's available in the gallery at
[awesome-streamlit.org](https://awesome-streamlit.org).
"""
from typing import Dict
import streamlit as st
from PIL import Image

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

FILE_TYPES = ["png", "jpg", "jpeg"]


@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}


def main():
    """Run this function to run the app"""
    static_store = get_static_store()

    st.sidebar.title("Multiple Image Stitching")

    st.info(__doc__)
    st.markdown(STYLE, unsafe_allow_html=True)
    result = st.file_uploader("Upload your images", type=FILE_TYPES)
    if result:
        # Process you file here
        value = result.getvalue()

        # And add it to the static_store if not already in
        if not value in static_store.values():
            static_store[result] = value
    else:
        static_store.clear()  # Hack to clear list if the user clears the cache and reloads the page
        st.info("Upload one or more images of type ." + ", ".join(FILE_TYPES))

    if st.button("Clear file list"):
        static_store.clear()
    if st.checkbox("Show file list?", True):
        st.write(list(static_store.keys()))
    if st.checkbox("Show content of files?"):
        images = []
        for value in static_store.values():
            images.append(value)
        if images != []:
            st.image(images, width=int(650/len(images)), caption=[str(i) for i in range(len(images))])

main()
