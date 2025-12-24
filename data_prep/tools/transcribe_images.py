import csv
import re
import tkinter as tk
from pathlib import Path
from typing import Union

from PIL import Image, ImageTk

# Ex:  = Path("C:/Images")
image_directory = None
csv_path = None

if image_directory is None or csv_path is None:
    raise ValueError("Please open the script and set 'image_directory' and 'csv_path' before running.")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def natural_key(name: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def norm(p: Union[str, Path]) -> str:
    return Path(p).as_posix()


transcriptions = {}
csv_ordered_filenames = []
if csv_path.exists():
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for filename, text in reader:
            if "/" in filename or "\\" in filename:
                key = norm(filename)
            else:
                key = norm(image_directory / filename)
            transcriptions[key] = text
            csv_ordered_filenames.append(filename)

available_images = {
    p.name: p
    for p in sorted(
        (p for p in image_directory.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=lambda p: natural_key(p.name)
    )
}

images = []
image_keys = []
for filename in csv_ordered_filenames:
    if filename in available_images:
        full_path = available_images[filename]
        images.append(full_path)
        image_keys.append(norm(full_path))

for filename, path in available_images.items():
    if filename not in csv_ordered_filenames:
        images.append(path)
        key = norm(path)
        image_keys.append(key)
        transcriptions[key] = ""


def save_csv():
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "text"])
        written = set()
        for k in image_keys:
            filename_only = Path(k).name
            w.writerow([filename_only, transcriptions[k]])
            written.add(k)
        for k in (k for k in transcriptions.keys() if k not in written):
            filename_only = Path(k).name
            w.writerow([filename_only, transcriptions[k]])


index = 0


def show_image():
    key = image_keys[index]
    img = Image.open(images[index])
    img.thumbnail((800, 800))
    tk_img = ImageTk.PhotoImage(img)
    image_label.config(image=tk_img)
    image_label.image = tk_img
    text_entry.delete("1.0", tk.END)
    text_entry.insert(tk.END, transcriptions.get(key, ""))
    text_entry.tag_add("center", "1.0", "end")
    root.title(f"{images[index].name} ({index+1}/{len(images)})")


def _stash_current():
    key = image_keys[index]
    transcriptions[key] = text_entry.get("1.0", tk.END).strip()


def next_image(event=None):
    global index
    _stash_current()
    index = (index + 1) % len(images)
    save_csv()
    show_image()


def prev_image(event=None):
    global index
    _stash_current()
    index = (index - 1) % len(images)
    save_csv()
    show_image()


root = tk.Tk()
root.geometry("800x160")

image_label = tk.Label(root)
image_label.pack()

text_entry = tk.Text(root, height=5, wrap="word")
text_entry.tag_configure("center", justify="center")
text_entry.tag_add("center", "1.0", "end")
text_entry.pack(fill="x", pady=(10, 0))

root.bind("<Right>", next_image)
root.bind("<Left>", prev_image)
root.bind("<Return>", next_image)

show_image()
root.mainloop()
