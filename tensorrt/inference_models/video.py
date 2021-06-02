import cv2
import numpy as np


def read_frames(video_path, fidxs=None, grayscale=True):
    """Read frames from a video file.
    
    Args:
        video_path: Path to MP4
        fidxs: List of frame indices or None to read all frames (default: None)
        grayscale: Keep only one channel of the images (default: True)
    
    Returns:
        Loaded images in array of shape (n_frames, height, width, channels) and dtype uint8.
    """
    vr = cv2.VideoCapture(video_path)
    if fidxs is None:
        fidxs = np.arange(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for fidx in fidxs:
        vr.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        img = vr.read()[1]
        if grayscale:
            img = img[:, :, [0]]
        else:
            img = img[..., ::-1]
        frames.append(img)
    return np.stack(frames, axis=0)


def save_frames(frames: np.ndarray, filename: str, fidxs=None, chunk_size=1000):
    """Save frames to MP4."""
    import skvideo.io
    writer = skvideo.io.FFmpegWriter(
        filename,
        inputdict={"-r": "25"},
        outputdict={
            "-c:v": "libx264",
            "-preset": "superfast", # seekability
            "-g": "1", # grouping keyframe interval
            "-framerate": "25",
            "-crf": "20",  # compression rate
            "-pix_fmt": "yuv420p",
        },
    )
    if fidxs is None:
        fidxs = np.arange(len(frames))
    for inds in np.array_split(fidxs, int(np.ceil(len(fidxs) / chunk_size))):
        writer.writeFrame(frames[inds])
    writer.close()
