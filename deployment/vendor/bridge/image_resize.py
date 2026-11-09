"""Opt-in training-equivalent image resize; never changes actions or state."""
from PIL import Image


def prepare_resize_input(obs, preprocessor, mode):
    if mode == "bilinear":
        return obs
    if mode != "lanczos":
        raise ValueError("Unsupported image resize mode: " + mode)
    if not preprocessor.multiview or (preprocessor.img_height, preprocessor.img_width) != (384, 320):
        raise ValueError("LANCZOS path requires the audited 384x320 three-view layout")
    # The training reader decodes each view to these slots with LANCZOS.
    # Stock assembly then sees exact-size views and does not resample them.
    sizes = {"head_camera": (320, 256), "left_wrist_camera": (160, 128),
             "right_wrist_camera": (160, 128)}
    images = {}
    for role, size in sizes.items():
        image = obs["images"][role]
        if not isinstance(image, Image.Image):
            raise TypeError("Resize input must be decoded PIL images")
        images[role] = image.resize(size, Image.Resampling.LANCZOS)
    return {**obs, "images": images}
