import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


class LineDetection:
    def __init__(self, root):
        self.root = root
        self.root.title("Line Detection Tool")
        self.root.geometry("1100x1000")

        self.original_image = None
        self.processed_image = None
        self.edges_image = None

        self.original_photo = None
        self.processed_photo = None
        self.edges_photo = None

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
        # Preprocessing
        preprocess_frame = ttk.LabelFrame(parent, text="Preprocessing")
        preprocess_frame.pack(fill=tk.X, pady=5)

        ttk.Label(preprocess_frame, text="Blur Kernel Size:").pack()
        self.blur_kernel = tk.IntVar(value=5)
        blur_scale = ttk.Scale(preprocess_frame, from_=1, to=15, variable=self.blur_kernel, orient=tk.HORIZONTAL)
        blur_scale.pack(fill=tk.X)
        blur_scale.bind("<ButtonRelease-1>", self.on_parameter_change)
        self.blur_kernel_label = ttk.Label(preprocess_frame, text="5")
        self.blur_kernel_label.pack()

        # Canny Edge Detection
        canny_frame = ttk.LabelFrame(parent, text="Canny Edge Detection")
        canny_frame.pack(fill=tk.X, pady=5)

        ttk.Label(canny_frame, text="Low Threshold:").pack()
        self.canny_low = tk.IntVar(value=0)
        canny_low_scale = ttk.Scale(canny_frame, from_=0, to=300, variable=self.canny_low, orient=tk.HORIZONTAL)
        canny_low_scale.pack(fill=tk.X)
        canny_low_scale.bind("<ButtonRelease-1>", self.on_parameter_change)
        self.canny_low_label = ttk.Label(canny_frame, text="0")
        self.canny_low_label.pack()

        ttk.Label(canny_frame, text="High Threshold:").pack()
        self.canny_high = tk.IntVar(value=300)
        canny_high_scale = ttk.Scale(canny_frame, from_=0, to=300, variable=self.canny_high, orient=tk.HORIZONTAL)
        canny_high_scale.pack(fill=tk.X)
        canny_high_scale.bind("<ButtonRelease-1>", self.on_parameter_change)
        self.canny_high_label = ttk.Label(canny_frame, text="300")
        self.canny_high_label.pack()

        ttk.Label(canny_frame, text="Morph Kernel Size:").pack()
        self.morph_kernel = tk.IntVar(value=1)
        morph_scale = ttk.Scale(canny_frame, from_=1, to=10, variable=self.morph_kernel, orient=tk.HORIZONTAL)
        morph_scale.pack(fill=tk.X)
        morph_scale.bind("<ButtonRelease-1>", self.on_parameter_change)
        self.morph_kernel_label = ttk.Label(canny_frame, text="1")
        self.morph_kernel_label.pack()

        # Hough Transform
        hough_frame = ttk.LabelFrame(parent, text="Hough Line Detection")
        hough_frame.pack(fill=tk.X, pady=5)

        ttk.Label(hough_frame, text="Threshold:").pack()
        self.hough_threshold = tk.IntVar(value=100)
        thresh_scale = ttk.Scale(hough_frame, from_=10, to=500, variable=self.hough_threshold, orient=tk.HORIZONTAL)
        thresh_scale.pack(fill=tk.X)
        thresh_scale.bind("<ButtonRelease-1>", self.on_parameter_change)
        self.hough_threshold_label = ttk.Label(hough_frame, text="100")
        self.hough_threshold_label.pack()

        ttk.Label(hough_frame, text="Min Line Length:").pack()
        self.min_line_length = tk.IntVar(value=100)
        length_scale = ttk.Scale(hough_frame, from_=10, to=6000, variable=self.min_line_length, orient=tk.HORIZONTAL)
        length_scale.pack(fill=tk.X)
        length_scale.bind("<ButtonRelease-1>", self.on_parameter_change)
        self.min_line_length_label = ttk.Label(hough_frame, text="100")
        self.min_line_length_label.pack()

        ttk.Label(hough_frame, text="Max Line Gap:").pack()
        self.max_line_gap = tk.IntVar(value=1)
        gap_scale = ttk.Scale(hough_frame, from_=1, to=200, variable=self.max_line_gap, orient=tk.HORIZONTAL)
        gap_scale.pack(fill=tk.X)
        gap_scale.bind("<ButtonRelease-1>", self.on_parameter_change)
        self.max_line_gap_label = ttk.Label(hough_frame, text="1")
        self.max_line_gap_label.pack()

        # Edge Density
        density_frame = ttk.LabelFrame(parent, text="Edge Density Analysis")
        density_frame.pack(fill=tk.X, pady=5)

        ttk.Label(density_frame, text="Strip Height:").pack()
        self.strip_height = tk.IntVar(value=21)
        strip_scale = ttk.Scale(density_frame, from_=5, to=50, variable=self.strip_height, orient=tk.HORIZONTAL)
        strip_scale.pack(fill=tk.X)
        strip_scale.bind("<ButtonRelease-1>", self.on_parameter_change)
        self.strip_height_label = ttk.Label(density_frame, text="21")
        self.strip_height_label.pack()

        ttk.Label(density_frame, text="Threshold Factor:").pack()
        self.density_threshold = tk.DoubleVar(value=0.5)
        density_scale = ttk.Scale(
            density_frame, from_=0.5, to=5.0, variable=self.density_threshold, orient=tk.HORIZONTAL
        )
        density_scale.pack(fill=tk.X)
        density_scale.bind("<ButtonRelease-1>", self.on_parameter_change)
        self.density_threshold_label = ttk.Label(density_frame, text="0.5")
        self.density_threshold_label.pack()

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

        processed_frame = ttk.Frame(self.notebook)
        self.notebook.add(processed_frame, text="Detected Lines")
        self.processed_canvas = tk.Canvas(processed_frame, bg="white")
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        self.bind_canvas_events(self.processed_canvas)

        edges_frame = ttk.Frame(self.notebook)
        self.notebook.add(edges_frame, text="Edges")
        self.edges_canvas = tk.Canvas(edges_frame, bg="white")
        self.edges_canvas.pack(fill=tk.BOTH, expand=True)
        self.bind_canvas_events(self.edges_canvas)

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

            self.process_image()
            self.update_parameters_display()
            self.root.after(100, self.fit_to_canvas)

    def process_image(self):
        if self.original_image is None:
            return

        params = self.get_current_parameters()
        img = self.original_image
        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_k = params["blur_kernel"]
        if blur_k % 2 == 0:
            blur_k += 1
        processed = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        edges = cv2.Canny(processed, params["canny_low"], params["canny_high"])
        kernel = np.ones((params["morph_kernel"], params["morph_kernel"]), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        self.edges_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        detections = []

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            params["hough_threshold"],
            minLineLength=params["min_line_length"],
            maxLineGap=params["max_line_gap"],
        )
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle <= 15 or angle >= 165:
                    detections.append({"center": max(y1, y2), "type": "hough", "coords": (x1, y1, x2, y2)})

        strip_height = params["strip_height"]
        densities = []
        for y in range(0, h - strip_height, 10):
            strip = edges[y : y + strip_height, :]
            density = np.sum(strip) / (strip_height * w)
            densities.append((y + strip_height // 2, density))

        dens_vals = [d[1] for d in densities]
        if dens_vals:
            threshold = np.mean(dens_vals) + params["density_threshold"] * np.std(dens_vals)
            for y, density in densities:
                if density > threshold:
                    detections.append({"center": y, "type": "edge_density"})

        self.processed_image = img.copy()

        for det in detections:
            if det["type"] == "hough" and "coords" in det:
                x1, y1, x2, y2 = det["coords"]
                cv2.line(self.processed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.line(self.processed_image, (0, det["center"]), (w, det["center"]), (255, 0, 255), 2)

        if detections:
            max_det = max(detections, key=lambda x: x["center"])
            cv2.line(self.processed_image, (0, max_det["center"]), (w, max_det["center"]), (0, 255, 0), 20)

    def on_parameter_change(self, *args):
        self.process_image()
        self.update_parameters_display()
        self.rebuild_photos()

    def on_tab_changed(self, event):
        current_tab = self.notebook.index(self.notebook.select())
        canvas_photo_pairs = [
            (self.original_canvas, self.original_photo),
            (self.processed_canvas, self.processed_photo),
            (self.edges_canvas, self.edges_photo),
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
        self.processed_photo = self.create_photo(self.processed_image)
        self.edges_photo = self.create_photo(self.edges_image)
        self.redraw_all_canvases()

    def create_photo(self, image):
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        new_width = max(1, int(image_pil.width * self.zoom_factor))
        new_height = max(1, int(image_pil.height * self.zoom_factor))
        image_pil = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return ImageTk.PhotoImage(image_pil)

    def redraw_all_canvases(self):
        self.draw_photo_on_canvas(self.original_photo, self.original_canvas)
        self.draw_photo_on_canvas(self.processed_photo, self.processed_canvas)
        self.draw_photo_on_canvas(self.edges_photo, self.edges_canvas)

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
            "blur_kernel": self.blur_kernel.get(),
            "canny_low": self.canny_low.get(),
            "canny_high": self.canny_high.get(),
            "morph_kernel": self.morph_kernel.get(),
            "hough_threshold": self.hough_threshold.get(),
            "min_line_length": self.min_line_length.get(),
            "max_line_gap": self.max_line_gap.get(),
            "strip_height": self.strip_height.get(),
            "density_threshold": round(self.density_threshold.get(), 2),
        }

    def update_parameters_display(self):
        params = self.get_current_parameters()

        self.params_text.delete(1.0, tk.END)

        display_text = "=== PARAMETERS ===\n\n"
        display_text += f"Blur Kernel: {params['blur_kernel']}\n"
        display_text += f"Canny: {params['canny_low']}, {params['canny_high']}\n"
        display_text += f"Morph Kernel: {params['morph_kernel']}\n"
        display_text += f"Hough Threshold: {params['hough_threshold']}\n"
        display_text += f"Min Line Length: {params['min_line_length']}\n"
        display_text += f"Max Line Gap: {params['max_line_gap']}\n"
        display_text += f"Strip Height: {params['strip_height']}\n"
        display_text += f"Density Threshold: {params['density_threshold']}\n"

        self.params_text.insert(1.0, display_text)

        self.blur_kernel_label.config(text=str(params["blur_kernel"]))
        self.canny_low_label.config(text=str(params["canny_low"]))
        self.canny_high_label.config(text=str(params["canny_high"]))
        self.morph_kernel_label.config(text=str(params["morph_kernel"]))
        self.hough_threshold_label.config(text=str(params["hough_threshold"]))
        self.min_line_length_label.config(text=str(params["min_line_length"]))
        self.max_line_gap_label.config(text=str(params["max_line_gap"]))
        self.strip_height_label.config(text=str(params["strip_height"]))
        self.density_threshold_label.config(text=str(params["density_threshold"]))

    def copy_params_to_clipboard(self):
        params = self.get_current_parameters()

        clipboard_content = "# Line Detection Parameters\n\n"
        clipboard_content += json.dumps(params, indent=4)

        self.root.clipboard_clear()
        self.root.clipboard_append(clipboard_content)
        messagebox.showinfo("Success", "Parameters copied to clipboard!")

    def save_parameters(self):
        params = self.get_current_parameters()

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

                self.blur_kernel.set(params["blur_kernel"])
                self.canny_low.set(params["canny_low"])
                self.canny_high.set(params["canny_high"])
                self.morph_kernel.set(params["morph_kernel"])
                self.hough_threshold.set(params["hough_threshold"])
                self.min_line_length.set(params["min_line_length"])
                self.max_line_gap.set(params["max_line_gap"])
                self.strip_height.set(params["strip_height"])
                self.density_threshold.set(params["density_threshold"])

                self.on_parameter_change()
                messagebox.showinfo("Success", f"Parameters loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load parameters: {str(e)}")


def main():
    root = tk.Tk()
    app = LineDetection(root)
    root.bind("<ButtonRelease-1>", app.end_pan)
    root.mainloop()


if __name__ == "__main__":
    main()
