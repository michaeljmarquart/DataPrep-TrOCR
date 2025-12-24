import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.filters import threshold_sauvola


class OCRPreprocessing:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV OCR Preprocessing Tool")
        self.root.geometry("1100x1000")

        self.original_image = None
        self.processed_image = None

        self.original_photo = None
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
        ttk.Button(control_frame, text="Save Processed", command=self.save_image).pack(pady=5)

        self.create_controls(control_frame)
        self.create_parameters_display(params_frame)
        self.create_image_display(image_frame)

    def create_controls(self, parent):
        blur_frame = ttk.LabelFrame(parent, text="Gaussian Blur")
        blur_frame.pack(fill=tk.X, pady=5)

        self.blur_enabled = tk.BooleanVar()
        ttk.Checkbutton(blur_frame, text="Enable", variable=self.blur_enabled, command=self.on_param_change).pack()

        ttk.Label(blur_frame, text="Kernel Size:").pack()
        self.blur_kernel = tk.IntVar(value=5)

        blur_scale = ttk.Scale(
            blur_frame,
            from_=1,
            to=15,
            variable=self.blur_kernel,
            orient=tk.HORIZONTAL,
        )
        blur_scale.pack(fill=tk.X)
        blur_scale.bind("<ButtonRelease-1>", self.on_param_change)

        thresh_frame = ttk.LabelFrame(parent, text="Threshold")
        thresh_frame.pack(fill=tk.X, pady=5)

        self.thresh_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(thresh_frame, text="Enable", variable=self.thresh_enabled, command=self.on_thresh_toggle).pack()

        self.thresh_type = tk.StringVar(value="BINARY")
        ttk.Label(thresh_frame, text="Type:").pack()

        thresh_combo = ttk.Combobox(
            thresh_frame,
            textvariable=self.thresh_type,
            values=[
                "BINARY",
                "BINARY_INV",
                "OTSU",
                "ADAPTIVE_MEAN",
                "ADAPTIVE_GAUSSIAN",
                "ADAPTIVE_MEAN_INV",
                "ADAPTIVE_GAUSSIAN_INV",
            ],
        )
        thresh_combo.pack(fill=tk.X)
        thresh_combo.bind("<<ComboboxSelected>>", self.on_param_change)

        ttk.Label(thresh_frame, text="Threshold Value:").pack()
        self.thresh_value = tk.IntVar(value=127)

        thresh_scale = ttk.Scale(thresh_frame, from_=0, to=255, variable=self.thresh_value, orient=tk.HORIZONTAL)
        thresh_scale.pack(fill=tk.X)
        thresh_scale.bind("<ButtonRelease-1>", self.on_param_change)

        sauvola_frame = ttk.LabelFrame(parent, text="Sauvola Threshold")
        sauvola_frame.pack(fill=tk.X, pady=5)

        self.sauvola_enabled = tk.BooleanVar()
        ttk.Checkbutton(
            sauvola_frame, text="Enable", variable=self.sauvola_enabled, command=self.on_sauvola_toggle
        ).pack()

        ttk.Label(sauvola_frame, text="Window Size:").pack()
        self.sauvola_window = tk.IntVar(value=25)
        sauvola_window_scale = ttk.Scale(
            sauvola_frame, from_=3, to=51, variable=self.sauvola_window, orient=tk.HORIZONTAL
        )
        sauvola_window_scale.pack(fill=tk.X)
        sauvola_window_scale.bind("<ButtonRelease-1>", self.on_param_change)

        ttk.Label(sauvola_frame, text="K:").pack()
        self.sauvola_k = tk.DoubleVar(value=0.2)

        sauvola_k_scale = ttk.Scale(sauvola_frame, from_=0.0, to=1.0, variable=self.sauvola_k, orient=tk.HORIZONTAL)
        sauvola_k_scale.pack(fill=tk.X)
        sauvola_k_scale.bind("<ButtonRelease-1>", self.on_param_change)

        morph_frame = ttk.LabelFrame(parent, text="Morphological Operations")
        morph_frame.pack(fill=tk.X, pady=5)

        self.morph_enabled = tk.BooleanVar()
        ttk.Checkbutton(morph_frame, text="Enable", variable=self.morph_enabled, command=self.on_param_change).pack()

        self.morph_operation = tk.StringVar(value="CLOSE")
        ttk.Label(morph_frame, text="Operation:").pack()

        morph_combo = ttk.Combobox(
            morph_frame, textvariable=self.morph_operation, values=["OPEN", "CLOSE", "ERODE", "DILATE"]
        )
        morph_combo.pack(fill=tk.X)
        morph_combo.bind("<<ComboboxSelected>>", self.on_param_change)

        ttk.Label(morph_frame, text="Kernel Size:").pack()
        self.morph_kernel = tk.IntVar(value=3)

        morph_scale = ttk.Scale(morph_frame, from_=1, to=10, variable=self.morph_kernel, orient=tk.HORIZONTAL)
        morph_scale.pack(fill=tk.X)
        morph_scale.bind("<ButtonRelease-1>", self.on_param_change)

        noise_frame = ttk.LabelFrame(parent, text="Noise Reduction")
        noise_frame.pack(fill=tk.X, pady=5)

        self.denoise_enabled = tk.BooleanVar()
        ttk.Checkbutton(noise_frame, text="Enable", variable=self.denoise_enabled, command=self.on_param_change).pack()

        ttk.Label(noise_frame, text="Filter Strength:").pack()
        self.denoise_strength = tk.IntVar(value=10)

        denoise_scale = ttk.Scale(noise_frame, from_=1, to=30, variable=self.denoise_strength, orient=tk.HORIZONTAL)
        denoise_scale.pack(fill=tk.X)
        denoise_scale.bind("<ButtonRelease-1>", self.on_param_change)

    def create_parameters_display(self, parent):
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.params_text = tk.Text(text_frame, height=15, width=35, font=("Consolas", 9))
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
        self.notebook.add(processed_frame, text="Processed")
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

            self.process_image()
            self.update_parameters_display()
            self.root.after(100, self.fit_to_canvas)

    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Processed Image",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")],
        )

        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                messagebox.showinfo("Success", f"Image saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save image: {str(e)}")

    def process_image(self):
        if self.original_image is None:
            return

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        if self.blur_enabled.get():
            kernel_size = self.blur_kernel.get()
            if kernel_size % 2 == 0:
                kernel_size += 1
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        if self.denoise_enabled.get():
            gray = cv2.fastNlMeansDenoising(gray, None, self.denoise_strength.get(), 7, 21)

        if self.thresh_enabled.get():
            thresh_type = self.thresh_type.get()

            if thresh_type == "BINARY":
                _, gray = cv2.threshold(gray, self.thresh_value.get(), 255, cv2.THRESH_BINARY)
            elif thresh_type == "BINARY_INV":
                _, gray = cv2.threshold(gray, self.thresh_value.get(), 255, cv2.THRESH_BINARY_INV)
            elif thresh_type == "OTSU":
                _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif thresh_type == "ADAPTIVE_MEAN":
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            elif thresh_type == "ADAPTIVE_GAUSSIAN":
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            elif thresh_type == "ADAPTIVE_MEAN_INV":
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            elif thresh_type == "ADAPTIVE_GAUSSIAN_INV":
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        if self.sauvola_enabled.get():
            window_size = self.sauvola_window.get()
            if window_size % 2 == 0:
                window_size += 1
            thresh_sauvola = threshold_sauvola(gray, window_size=window_size, k=self.sauvola_k.get())
            gray = (gray > thresh_sauvola).astype(np.uint8) * 255

        if self.morph_enabled.get():
            kernel_size = self.morph_kernel.get()
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            operation = self.morph_operation.get()
            if operation == "OPEN":
                gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            elif operation == "CLOSE":
                gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            elif operation == "ERODE":
                gray = cv2.erode(gray, kernel, iterations=1)
            elif operation == "DILATE":
                gray = cv2.dilate(gray, kernel, iterations=1)

        self.processed_image = gray

    def on_thresh_toggle(self, *args):
        if self.thresh_enabled.get():
            self.sauvola_enabled.set(False)
        self.on_param_change()

    def on_sauvola_toggle(self, *args):
        if self.sauvola_enabled.get():
            self.thresh_enabled.set(False)
        self.on_param_change()

    def on_param_change(self, *args):
        self.process_image()
        self.update_parameters_display()
        self.rebuild_photos()

    def on_tab_changed(self, event):
        current_tab = self.notebook.index(self.notebook.select())
        canvas_photo_pairs = [
            (self.original_canvas, self.original_photo),
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
        params = {
            "gaussian_blur": {"enabled": self.blur_enabled.get(), "kernel_size": self.blur_kernel.get()},
            "threshold": {
                "enabled": self.thresh_enabled.get(),
                "type": self.thresh_type.get(),
                "value": self.thresh_value.get(),
            },
            "sauvola_threshold": {
                "enabled": self.sauvola_enabled.get(),
                "window_size": self.sauvola_window.get(),
                "k": self.sauvola_k.get(),
            },
            "morphological": {
                "enabled": self.morph_enabled.get(),
                "operation": self.morph_operation.get(),
                "kernel_size": self.morph_kernel.get(),
            },
            "noise_reduction": {"enabled": self.denoise_enabled.get(), "strength": self.denoise_strength.get()},
        }
        return params

    def update_parameters_display(self):
        params = self.get_current_parameters()

        self.params_text.delete(1.0, tk.END)

        display_text = "=== CURRENT PARAMETERS ===\n\n"

        display_text += f"GAUSSIAN BLUR:\n"
        display_text += f"  Enabled: {params['gaussian_blur']['enabled']}\n"
        if params["gaussian_blur"]["enabled"]:
            kernel = params["gaussian_blur"]["kernel_size"]
            if kernel % 2 == 0:
                kernel += 1
            display_text += f"  Kernel Size: {kernel}\n"
        display_text += "\n"

        display_text += f"THRESHOLD:\n"
        display_text += f"  Enabled: {params['threshold']['enabled']}\n"
        if params["threshold"]["enabled"]:
            display_text += f"  Type: {params['threshold']['type']}\n"
            if params["threshold"]["type"] in ["BINARY", "BINARY_INV"]:
                display_text += f"  Value: {params['threshold']['value']}\n"
        display_text += "\n"

        display_text += f"SAUVOLA THRESHOLD:\n"
        display_text += f"  Enabled: {params['sauvola_threshold']['enabled']}\n"
        if params["sauvola_threshold"]["enabled"]:
            window = params["sauvola_threshold"]["window_size"]
            if window % 2 == 0:
                window += 1
            display_text += f"  Window Size: {window}\n"
            display_text += f"  K: {params['sauvola_threshold']['k']:.2f}\n"
        display_text += "\n"

        display_text += f"MORPHOLOGICAL:\n"
        display_text += f"  Enabled: {params['morphological']['enabled']}\n"
        if params["morphological"]["enabled"]:
            display_text += f"  Operation: {params['morphological']['operation']}\n"
            display_text += f"  Kernel Size: {params['morphological']['kernel_size']}\n"
        display_text += "\n"

        display_text += f"NOISE REDUCTION:\n"
        display_text += f"  Enabled: {params['noise_reduction']['enabled']}\n"
        if params["noise_reduction"]["enabled"]:
            display_text += f"  Strength: {params['noise_reduction']['strength']}\n"
        display_text += "\n"

        display_text += "=== PYTHON CODE ===\n\n"
        display_text += self.generate_code_snippet(params)

        self.params_text.insert(1.0, display_text)

    def generate_code_snippet(self, params):
        code_lines = ["import cv2", "import numpy as np"]

        if params["sauvola_threshold"]["enabled"]:
            code_lines.append("from skimage.filters import threshold_sauvola")

        code_lines.extend(
            [
                "",
                "def preprocess_image(image_path):",
                "    image = cv2.imread(image_path)",
                "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)",
                "",
            ]
        )

        if params["gaussian_blur"]["enabled"]:
            kernel = params["gaussian_blur"]["kernel_size"]
            if kernel % 2 == 0:
                kernel += 1
            code_lines.extend(
                ["    # Gaussian Blur", f"    gray = cv2.GaussianBlur(gray, ({kernel}, {kernel}), 0)", ""]
            )

        if params["noise_reduction"]["enabled"]:
            strength = params["noise_reduction"]["strength"]
            code_lines.extend(
                ["    # Noise Reduction", f"    gray = cv2.fastNlMeansDenoising(gray, None, {strength}, 7, 21)", ""]
            )

        if params["threshold"]["enabled"]:
            thresh_type = params["threshold"]["type"]
            thresh_val = params["threshold"]["value"]

            code_lines.append("    # Thresholding")
            if thresh_type == "BINARY":
                code_lines.append(f"    _, gray = cv2.threshold(gray, {thresh_val}, 255, cv2.THRESH_BINARY)")
            elif thresh_type == "BINARY_INV":
                code_lines.append(f"    _, gray = cv2.threshold(gray, {thresh_val}, 255, cv2.THRESH_BINARY_INV)")
            elif thresh_type == "OTSU":
                code_lines.append("    _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)")
            elif thresh_type == "ADAPTIVE_MEAN":
                code_lines.append(
                    "    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)"
                )
            elif thresh_type == "ADAPTIVE_GAUSSIAN":
                code_lines.append(
                    "    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)"
                )
            elif thresh_type == "ADAPTIVE_MEAN_INV":
                code_lines.append(
                    "    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)"
                )
            elif thresh_type == "ADAPTIVE_GAUSSIAN_INV":
                code_lines.append(
                    "    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)"
                )
            code_lines.append("")

        if params["sauvola_threshold"]["enabled"]:
            window = params["sauvola_threshold"]["window_size"]
            k = params["sauvola_threshold"]["k"]
            # Check for odd #
            if window % 2 == 0:
                window += 1

            code_lines.extend(
                [
                    "    # Sauvola Thresholding",
                    f"    thresh_sauvola = threshold_sauvola(gray, window_size={window}, k={k})",
                    "    gray = (gray > thresh_sauvola).astype(np.uint8) * 255",
                    "",
                ]
            )

        if params["morphological"]["enabled"]:
            operation = params["morphological"]["operation"]
            kernel_size = params["morphological"]["kernel_size"]
            code_lines.extend(
                ["    # Morphological Operations", f"    kernel = np.ones(({kernel_size}, {kernel_size}), np.uint8)"]
            )

            if operation == "OPEN":
                code_lines.append("    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)")
            elif operation == "CLOSE":
                code_lines.append("    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)")
            elif operation == "ERODE":
                code_lines.append("    gray = cv2.erode(gray, kernel, iterations=1)")
            elif operation == "DILATE":
                code_lines.append("    gray = cv2.dilate(gray, kernel, iterations=1)")
            code_lines.append("")

        code_lines.extend(
            [
                "    return gray",
                "",
                "# Usage:",
                "# processed_image = preprocess_image('your_image.jpg')",
                "# cv2.imwrite('processed_image.jpg', processed_image)",
            ]
        )

        return "\n".join(code_lines)

    def copy_params_to_clipboard(self):
        params = self.get_current_parameters()

        clipboard_content = "# OCR Preprocessing Parameters\n\n"
        clipboard_content += "# JSON Format:\n"
        clipboard_content += json.dumps(params, indent=2)
        clipboard_content += "\n\n# Python Code:\n"
        clipboard_content += self.generate_code_snippet(params)

        self.root.clipboard_clear()
        self.root.clipboard_append(clipboard_content)
        messagebox.showinfo("Success", "Parameters and code copied to clipboard!")

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

                self.blur_enabled.set(params["gaussian_blur"]["enabled"])
                self.blur_kernel.set(params["gaussian_blur"]["kernel_size"])

                self.thresh_enabled.set(params["threshold"]["enabled"])
                self.thresh_type.set(params["threshold"]["type"])
                self.thresh_value.set(params["threshold"]["value"])

                if "sauvola_threshold" in params:
                    self.sauvola_enabled.set(params["sauvola_threshold"]["enabled"])
                    self.sauvola_window.set(params["sauvola_threshold"]["window_size"])
                    self.sauvola_k.set(params["sauvola_threshold"]["k"])
                else:
                    self.sauvola_enabled.set(False)

                self.morph_enabled.set(params["morphological"]["enabled"])
                self.morph_operation.set(params["morphological"]["operation"])
                self.morph_kernel.set(params["morphological"]["kernel_size"])

                self.denoise_enabled.set(params["noise_reduction"]["enabled"])
                self.denoise_strength.set(params["noise_reduction"]["strength"])

                if self.thresh_enabled.get():
                    self.sauvola_enabled.set(False)
                elif self.sauvola_enabled.get():
                    self.thresh_enabled.set(False)

                self.on_param_change()

                messagebox.showinfo("Success", f"Parameters loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load parameters: {str(e)}")


def main():
    root = tk.Tk()
    app = OCRPreprocessing(root)
    root.bind("<ButtonRelease-1>", app.end_pan)
    root.mainloop()


if __name__ == "__main__":
    main()
