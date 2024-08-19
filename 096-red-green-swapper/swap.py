import sys
from PIL import Image
import imageio
import numpy as np
import os

def swap_channels(frame):
    img = Image.fromarray(frame)
    r, g, b = img.split()
    img = Image.merge("RGB", (g, r, b))
    return np.array(img)

def process_video(input_file, output_file):
    reader = imageio.get_reader(input_file)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(output_file, fps=fps)

    for frame in reader:
        modified_frame = swap_channels(frame)
        writer.append_data(modified_frame)

    writer.close()
    print(f"Swapped video saved as {output_file}")

def process_image(input_file, output_file):
    img = Image.open(input_file)
    r, g, b = img.split()
    img = Image.merge("RGB", (g, r, b))
    img.save(output_file)
    print(f"Swapped image saved as {output_file}")

if len(sys.argv) != 3:
    print("Usage: python swap_channels.py input_file output_file")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Determine if the input file is an image or video
file_extension = os.path.splitext(input_file)[1].lower()

if file_extension in ['.mp4', '.mov', '.avi', '.mkv']:
    process_video(input_file, output_file)
elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
    process_image(input_file, output_file)
else:
    print("Unsupported file type.")
    sys.exit(1)
