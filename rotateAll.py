# import subprocess
# from pathlib import Path

# def rotate_video_ffmpeg(input_path: str, output_path: str, clockwise_degrees: int = 90):
#     """
#     Rotate a single video using FFmpeg without black bars.
#     """
#     inp = Path(input_path)
#     out = Path(output_path)

#     if not inp.exists():
#         raise FileNotFoundError(f"Input not found: {inp}")

#     # Choose FFmpeg transpose filter
#     if clockwise_degrees == 90:
#         vf = "transpose=1"
#     elif clockwise_degrees == 270:
#         vf = "transpose=2"
#     elif clockwise_degrees == 180:
#         vf = "transpose=1,transpose=1"
#     else:
#         raise ValueError("Only 90, 180, or 270 degrees supported.")

#     # FFmpeg command
#     cmd = [
#         "ffmpeg",
#         "-y",
#         "-i", str(inp),
#         "-vf", vf,
#         "-c:a", "copy",
#         str(out)
#     ]

#     print(f"🎞 Rotating: {inp.name}")
#     subprocess.run(cmd, check=True)
#     print(f"✅ Saved rotated video as: {out.name}\n")


# def rotate_all_videos_in_folder(input_folder: str, output_folder: str, clockwise_degrees: int = 90):
#     """
#     Rotates all .mp4 videos from input_folder by the given angle
#     and saves them into output_folder with '_rotated' appended.
#     """
#     input_dir = Path(input_folder)
#     output_dir = Path(output_folder)
#     output_dir.mkdir(parents=True, exist_ok=True)  # create if not exists

#     if not input_dir.exists():
#         raise FileNotFoundError(f"Input folder not found: {input_dir}")

#     for video in input_dir.glob("*.mp4"):
#         output_file = output_dir / f"{video.stem}_rotated.mp4"
#         rotate_video_ffmpeg(video, output_file, clockwise_degrees)

#     print("🎯 All videos processed successfully!")


# # 🧩 Example usage
# rotate_all_videos_in_folder(
#     input_folder=r"C:\Users\prana\Downloads\Wrist_Flexion_Stretch",
#     output_folder=r"C:\Users\prana\Downloads\Wrist_Flexion_Stretch_rotated",
#     clockwise_degrees=90  # rotate each video by 90° clockwise
# )


import subprocess
from pathlib import Path

def mirror_video_ffmpeg(input_path: str, output_path: str):
    """
    Mirrors (horizontally flips) a single video using FFmpeg.
    """
    inp = Path(input_path)
    out = Path(output_path)

    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {inp}")

    # FFmpeg command for horizontal flip
    cmd = [
        "ffmpeg",
        "-y",                # overwrite output if exists
        "-i", str(inp),      # input file
        "-vf", "hflip",      # horizontal flip
        "-c:a", "copy",      # keep audio as-is
        str(out)             # output file
    ]

    print(f"🎞 Mirroring: {inp.name}")
    subprocess.run(cmd, check=True)
    print(f"✅ Saved mirrored video as: {out.name}\n")


def mirror_all_original_videos(input_folder: str, output_folder: str):
    """
    Flips all .mp4 videos in the input folder horizontally,
    saves them to output_folder, and appends '_mirrored' to filenames.
    """
    input_dir = Path(input_folder)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)  # auto-create if not exists

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    # Loop through all MP4 videos
    for video in input_dir.glob("*.mp4"):
        output_file = output_dir / f"{video.stem}_mirrored.mp4"
        mirror_video_ffmpeg(video, output_file)

    print("🎯 All original videos mirrored successfully!")


# 🧠 Example usage
mirror_all_original_videos(
    input_folder=r"C:\Users\prana\Downloads\Wrist_Flexion_Stretch",   # original videos
    output_folder=r"C:\Users\prana\Downloads\Wrist_Flexion_Stretch_mirrored"  # new mirrored folder
)

