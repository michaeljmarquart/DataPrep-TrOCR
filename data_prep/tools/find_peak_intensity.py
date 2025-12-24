import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.signal import find_peaks


class FindPeak:
    def __init__(self, root):
        self.root = root
        self.root.title("Peak Detection Tool")
        self.root.geometry("1100x1000")

        self.original_image = None
        self.preprocessed_image = None
        self.processed_image = None

        self.original_photo = None
        self.preprocessed_photo = None
        self.processed_photo = None

        self.zoom_factor = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.image_x_offset = 0
        self.image_y_offset = 0
        self.zoom_job = None
        self.setup_gui()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        control_frame = ttk.Frame(left_panel)
        control_frame.pack(fill=tk.BOTH, expand=True)

        params_frame = ttk.LabelFrame(left_panel, text="Current Parameters")
        params_frame.pack(fill=tk.X, pady=(10, 0))

        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(pady=5)

        self.create_controls(control_frame)
        self.create_parameters_display(params_frame)
        self.create_image_display(image_frame)

    def create_controls(self, parent):
        preprocess_frame = ttk.LabelFrame(parent, text="Preprocessing")
        preprocess_frame.pack(fill=tk.X, pady=5)

        ttk.Label(preprocess_frame, text="Denoise Strength:").pack()
        self.denoise_strength = tk.IntVar(value=10)
        denoise_scale = ttk.Scale(
            preprocess_frame, from_=0, to=30, variable=self.denoise_strength, orient=tk.HORIZONTAL
        )
        denoise_scale.pack(fill=tk.X)
        denoise_scale.bind("<ButtonRelease-1>", self.on_preprocess_change)
        self.denoise_strength_label = ttk.Label(preprocess_frame, text="10")
        self.denoise_strength_label.pack()

        ttk.Label(preprocess_frame, text="Threshold:").pack()
        self.threshold_value = tk.IntVar(value=234)
        threshold_scale = ttk.Scale(
            preprocess_frame, from_=0, to=255, variable=self.threshold_value, orient=tk.HORIZONTAL
        )
        threshold_scale.pack(fill=tk.X)
        threshold_scale.bind("<ButtonRelease-1>", self.on_preprocess_change)
        self.threshold_value_label = ttk.Label(preprocess_frame, text="234")
        self.threshold_value_label.pack()

        row_frame = ttk.LabelFrame(parent, text="Row Detection")
        row_frame.pack(fill=tk.X, pady=5)

        ttk.Label(row_frame, text="Min Height:").pack()
        self.row_height = tk.IntVar(value=103)
        row_height_scale = ttk.Scale(row_frame, from_=10, to=500, variable=self.row_height, orient=tk.HORIZONTAL)
        row_height_scale.pack(fill=tk.X)
        row_height_scale.bind("<ButtonRelease-1>", self.on_detection_change)
        self.row_height_label = ttk.Label(row_frame, text="103")
        self.row_height_label.pack()

        ttk.Label(row_frame, text="Min Distance:").pack()
        self.row_distance = tk.IntVar(value=58)
        row_distance_scale = ttk.Scale(row_frame, from_=5, to=200, variable=self.row_distance, orient=tk.HORIZONTAL)
        row_distance_scale.pack(fill=tk.X)
        row_distance_scale.bind("<ButtonRelease-1>", self.on_detection_change)
        self.row_distance_label = ttk.Label(row_frame, text="58")
        self.row_distance_label.pack()

        ttk.Label(row_frame, text="Prominence:").pack()
        self.row_prominence = tk.IntVar(value=106)
        row_prominence_scale = ttk.Scale(
            row_frame, from_=10, to=1000, variable=self.row_prominence, orient=tk.HORIZONTAL
        )
        row_prominence_scale.pack(fill=tk.X)
        row_prominence_scale.bind("<ButtonRelease-1>", self.on_detection_change)
        self.row_prominence_label = ttk.Label(row_frame, text="106")
        self.row_prominence_label.pack()

        col_frame = ttk.LabelFrame(parent, text="Column Detection")
        col_frame.pack(fill=tk.X, pady=5)

        ttk.Label(col_frame, text="Min Height:").pack()
        self.col_height = tk.IntVar(value=34)
        col_height_scale = ttk.Scale(col_frame, from_=10, to=500, variable=self.col_height, orient=tk.HORIZONTAL)
        col_height_scale.pack(fill=tk.X)
        col_height_scale.bind("<ButtonRelease-1>", self.on_detection_change)
        self.col_height_label = ttk.Label(col_frame, text="34")
        self.col_height_label.pack()

        ttk.Label(col_frame, text="Min Distance:").pack()
        self.col_distance = tk.IntVar(value=34)
        col_distance_scale = ttk.Scale(col_frame, from_=5, to=200, variable=self.col_distance, orient=tk.HORIZONTAL)
        col_distance_scale.pack(fill=tk.X)
        col_distance_scale.bind("<ButtonRelease-1>", self.on_detection_change)
        self.col_distance_label = ttk.Label(col_frame, text="34")
        self.col_distance_label.pack()

        ttk.Label(col_frame, text="Prominence:").pack()
        self.col_prominence = tk.IntVar(value=159)
        col_prominence_scale = ttk.Scale(
            col_frame, from_=10, to=1000, variable=self.col_prominence, orient=tk.HORIZONTAL
        )
        col_prominence_scale.pack(fill=tk.X)
        col_prominence_scale.bind("<ButtonRelease-1>", self.on_detection_change)
        self.col_prominence_label = ttk.Label(col_frame, text="159")
        self.col_prominence_label.pack()

    def create_parameters_display(self, parent):
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.params_text = tk.Text(text_frame, height=12, width=35, font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.params_text.yview)
        self.params_text.config(yscrollcommand=scrollbar.set)

        self.params_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(button_frame, text="Copy to Clipboard", command=self.copy_params_to_clipboard).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(button_frame, text="Save Parameters", command=self.save_parameters).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Load Parameters", command=self.load_parameters).pack(side=tk.LEFT, padx=2)

    def create_image_display(self, parent):
        zoom_frame = ttk.Frame(parent)
        zoom_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)

        self.zoom_label = ttk.Label(zoom_frame, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=10)

        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        original_frame = ttk.Frame(self.notebook)
        self.notebook.add(original_frame, text="Original")
        self.original_canvas = tk.Canvas(original_frame, bg="white")
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        self.bind_canvas_events(self.original_canvas)

        preprocessed_frame = ttk.Frame(self.notebook)
        self.notebook.add(preprocessed_frame, text="Preprocessed")
        self.preprocessed_canvas = tk.Canvas(preprocessed_frame, bg="white")
        self.preprocessed_canvas.pack(fill=tk.BOTH, expand=True)
        self.bind_canvas_events(self.preprocessed_canvas)

        processed_frame = ttk.Frame(self.notebook)
        self.notebook.add(processed_frame, text="Detected Lines")
        self.processed_canvas = tk.Canvas(processed_frame, bg="white")
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        self.bind_canvas_events(self.processed_canvas)

    def bind_canvas_events(self, canvas):
        canvas.bind("<Button-1>", self.start_pan)
        canvas.bind("<B1-Motion>", self.do_pan)
        canvas.bind("<MouseWheel>", self.mouse_zoom)
        canvas.bind("<Button-4>", self.mouse_zoom)
        canvas.bind("<Button-5>", self.mouse_zoom)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )

        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Error", "Could not load image")
                return

            self.image_x_offset = 0
            self.image_y_offset = 0

            self.preprocess_image()
            self.detect_peaks()
            self.update_parameters_display()
            self.root.after(100, self.fit_to_canvas)

    def preprocess_image(self):
        if self.original_image is None:
            return

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        denoise = self.denoise_strength.get()
        if denoise > 0:
            gray = cv2.fastNlMeansDenoising(gray, None, denoise, 7, 21)

        _, self.preprocessed_image = cv2.threshold(gray, self.threshold_value.get(), 255, cv2.THRESH_BINARY)

    def detect_peaks(self):
        if self.preprocessed_image is None or self.original_image is None:
            return

        self.processed_image = self.original_image.copy()
        img_height, img_width = self.processed_image.shape[:2]

        h_projection = np.mean(255 - self.preprocessed_image, axis=1)
        h_peaks, _ = find_peaks(
            h_projection,
            height=self.row_height.get(),
            distance=self.row_distance.get(),
            prominence=self.row_prominence.get(),
        )

        v_projection = np.mean(255 - self.preprocessed_image, axis=0)
        v_peaks, _ = find_peaks(
            v_projection,
            height=self.col_height.get(),
            distance=self.col_distance.get(),
            prominence=self.col_prominence.get(),
        )

        for peak in h_peaks:
            cv2.line(self.processed_image, (0, peak), (img_width, peak), (0, 0, 255), 2)

        for peak in v_peaks:
            cv2.line(self.processed_image, (peak, 0), (peak, img_height), (255, 0, 0), 2)

    def on_preprocess_change(self, *args):
        self.preprocess_image()
        self.detect_peaks()
        self.update_parameters_display()
        self.rebuild_photos()

    def on_detection_change(self, *args):
        self.detect_peaks()
        self.update_parameters_display()
        self.rebuild_photos()

    def on_tab_changed(self, event):
        current_tab = self.notebook.index(self.notebook.select())
        canvas_photo_pairs = [
            (self.original_canvas, self.original_photo),
            (self.preprocessed_canvas, self.preprocessed_photo),
            (self.processed_canvas, self.processed_photo),
        ]
        if current_tab < len(canvas_photo_pairs):
            canvas, photo = canvas_photo_pairs[current_tab]
            self.root.after(10, lambda: self.draw_photo_on_canvas(photo, canvas))

    def fit_to_canvas(self):
        if self.original_image is None:
            return

        canvas_width = self.original_canvas.winfo_width()
        canvas_height = self.original_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self.fit_to_canvas)
            return

        # Reset offsets to center the image
        self.image_x_offset = 0
        self.image_y_offset = 0

        img_height, img_width = self.original_image.shape[:2]
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.zoom_factor = min(scale_x, scale_y)

        self.zoom_label.config(text=f"{int(self.zoom_factor * 100)}%")
        self.rebuild_photos()

    def rebuild_photos(self):
        self.original_photo = self.create_photo(self.original_image)
        self.preprocessed_photo = self.create_photo(self.preprocessed_image)
        self.processed_photo = self.create_photo(self.processed_image)
        self.redraw_all_canvases()

    def create_photo(self, image):
        if image is None:
            return None

        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        new_width = max(1, int(image_pil.width * self.zoom_factor))
        new_height = max(1, int(image_pil.height * self.zoom_factor))
        image_pil = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return ImageTk.PhotoImage(image_pil)

    def redraw_all_canvases(self):
        self.draw_photo_on_canvas(self.original_photo, self.original_canvas)
        self.draw_photo_on_canvas(self.preprocessed_photo, self.preprocessed_canvas)
        self.draw_photo_on_canvas(self.processed_photo, self.processed_canvas)

    def draw_photo_on_canvas(self, photo, canvas):
        if photo is None:
            return

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        canvas.delete("all")
        canvas.create_image(
            canvas_width // 2 + self.image_x_offset,
            canvas_height // 2 + self.image_y_offset,
            image=photo,
            anchor=tk.CENTER,
        )

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor * 1.15, 10.0)
        self.zoom_label.config(text=f"{int(self.zoom_factor * 100)}%")
        self.schedule_rebuild()

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor / 1.15, 0.1)
        self.zoom_label.config(text=f"{int(self.zoom_factor * 100)}%")
        self.schedule_rebuild()

    def reset_zoom(self):
        if self.original_image is not None:
            self.fit_to_canvas()
        else:
            self.zoom_factor = 1.0
            self.image_x_offset = 0
            self.image_y_offset = 0
            self.zoom_label.config(text="100%")

    def mouse_zoom(self, event):
        """Handles mouse wheel events with debouncing."""
        if event.num == 4 or event.delta > 0:
            self.zoom_in()
        elif event.num == 5 or event.delta < 0:
            self.zoom_out()

    def schedule_rebuild(self):
        """Schedules rebuild_photos to run after a delay, canceling any pending requests."""
        if self.zoom_job:
            self.root.after_cancel(self.zoom_job)

        # Delay by 200ms. Adjust this value to make it more or less 'snappy'
        self.zoom_job = self.root.after(200, self.rebuild_photos)

    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        event.widget.config(cursor="fleur")

    def do_pan(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y

        self.image_x_offset += dx
        self.image_y_offset += dy

        self.pan_start_x = event.x
        self.pan_start_y = event.y

        self.redraw_all_canvases()

    def end_pan(self, event):
        event.widget.config(cursor="")

    def get_current_parameters(self):
        return {
            "row_min_height": self.row_height.get(),
            "row_min_distance": self.row_distance.get(),
            "row_prominence": self.row_prominence.get(),
            "col_min_height": self.col_height.get(),
            "col_min_distance": self.col_distance.get(),
            "col_prominence": self.col_prominence.get(),
        }

    def update_parameters_display(self):
        params = self.get_current_parameters()

        self.params_text.delete(1.0, tk.END)

        display_text = "=== PARAMETERS ===\n\n"
        display_text += "PREPROCESSING:\n"
        display_text += f"  Denoise: {self.denoise_strength.get()}\n"
        display_text += f"  Threshold: {self.threshold_value.get()}\n\n"
        display_text += "ROW DETECTION:\n"
        display_text += f"  Min Height: {params['row_min_height']}\n"
        display_text += f"  Min Distance: {params['row_min_distance']}\n"
        display_text += f"  Prominence: {params['row_prominence']}\n\n"
        display_text += "COLUMN DETECTION:\n"
        display_text += f"  Min Height: {params['col_min_height']}\n"
        display_text += f"  Min Distance: {params['col_min_distance']}\n"
        display_text += f"  Prominence: {params['col_prominence']}\n"

        self.params_text.insert(1.0, display_text)

        self.denoise_strength_label.config(text=str(self.denoise_strength.get()))
        self.threshold_value_label.config(text=str(self.threshold_value.get()))
        self.row_height_label.config(text=str(params["row_min_height"]))
        self.row_distance_label.config(text=str(params["row_min_distance"]))
        self.row_prominence_label.config(text=str(params["row_prominence"]))
        self.col_height_label.config(text=str(params["col_min_height"]))
        self.col_distance_label.config(text=str(params["col_min_distance"]))
        self.col_prominence_label.config(text=str(params["col_prominence"]))

    def copy_params_to_clipboard(self):
        params = self.get_current_parameters()

        clipboard_content = f"""PEAK_CONFIG = {{
    "denoise_strength": {self.denoise_strength.get()},
    "threshold": {self.threshold_value.get()},
    "row_min_height": {params["row_min_height"]},
    "row_min_distance": {params["row_min_distance"]},
    "row_prominence": {params["row_prominence"]},
    "col_min_height": {params["col_min_height"]},
    "col_min_distance": {params["col_min_distance"]},
    "col_prominence": {params["col_prominence"]},
}}"""

        self.root.clipboard_clear()
        self.root.clipboard_append(clipboard_content)
        messagebox.showinfo("Success", "Parameters copied to clipboard!")

    def save_parameters(self):
        params = self.get_current_parameters()
        params["denoise_strength"] = self.denoise_strength.get()
        params["threshold"] = self.threshold_value.get()

        file_path = filedialog.asksaveasfilename(
            title="Save Parameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if file_path:
            try:
                with open(file_path, "w") as f:
                    json.dump(params, f, indent=2)
                messagebox.showinfo("Success", f"Parameters saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save parameters: {str(e)}")

    def load_parameters(self):
        file_path = filedialog.askopenfilename(
            title="Load Parameters", filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, "r") as f:
                    params = json.load(f)

                self.denoise_strength.set(params.get("denoise_strength", 10))
                self.threshold_value.set(params.get("threshold", 234))
                self.row_height.set(params["row_min_height"])
                self.row_distance.set(params["row_min_distance"])
                self.row_prominence.set(params["row_prominence"])
                self.col_height.set(params["col_min_height"])
                self.col_distance.set(params["col_min_distance"])
                self.col_prominence.set(params["col_prominence"])

                self.on_preprocess_change()
                messagebox.showinfo("Success", f"Parameters loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load parameters: {str(e)}")


def main():
    root = tk.Tk()
    app = FindPeak(root)
    root.bind("<ButtonRelease-1>", app.end_pan)
    root.mainloop()


if __name__ == "__main__":
    main()
