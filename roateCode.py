import subprocess
from pathlib import Path

def rotate_video_ffmpeg(input_path: str, output_path: str, clockwise_degrees: int = 90):
    inp = Path(input_path)
    out = Path(output_path)

    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    if clockwise_degrees == 90:
        vf = "transpose=1"
    elif clockwise_degrees == 270:
        vf = "transpose=2"
    elif clockwise_degrees == 180:
        vf = "transpose=1,transpose=1"
    else:
        raise ValueError("Only 90, 180, or 270 degrees supported.")

    # Properly quote paths for Windows FFmpeg
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(inp),
        "-vf", vf,
        "-c:a", "copy",
        str(out)
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ Rotated video saved at: {out}")

# Example usage
rotate_video_ffmpeg(
    r"C:\Users\prana\Downloads\Wrist_Extension_Strengthening_1\wrist_extension_strengthening_1_good_4.mp4",
    r"C:\Users\prana\Downloads\roateFlipCode\output_rotated.mp4",
    90
)
