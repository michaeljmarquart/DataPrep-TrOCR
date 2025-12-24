import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


class FindTextBboxes:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Bbox Tool")
        self.root.geometry("1100x1000")

        self.original_image = None
        self.processed_image = None
        self.thresh_image = None
        self.dilated_image = None

        self.original_photo = None
        self.processed_photo = None
        self.thresh_photo = None
        self.dilated_photo = None

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
        thresh_frame = ttk.LabelFrame(parent, text="Threshold (Binary Inverse)")
        thresh_frame.pack(fill=tk.X, pady=5)

        ttk.Label(thresh_frame, text="Threshold Value:").pack()
        self.thresh_value = tk.IntVar(value=127)
        thresh_scale = ttk.Scale(thresh_frame, from_=0, to=255, variable=self.thresh_value, orient=tk.HORIZONTAL)
        thresh_scale.pack(fill=tk.X)
        thresh_scale.bind("<ButtonRelease-1>", self.on_parameter_change)

        self.thresh_value_label = ttk.Label(thresh_frame, text="127")
        self.thresh_value_label.pack()

        dilate_frame = ttk.LabelFrame(parent, text="Dilation")
        dilate_frame.pack(fill=tk.X, pady=5)

        ttk.Label(dilate_frame, text="Kernel Width:").pack()
        self.kernel_width = tk.IntVar(value=4)
        kernel_w_scale = ttk.Scale(dilate_frame, from_=1, to=20, variable=self.kernel_width, orient=tk.HORIZONTAL)
        kernel_w_scale.pack(fill=tk.X)
        kernel_w_scale.bind("<ButtonRelease-1>", self.on_parameter_change)

        self.kernel_width_label = ttk.Label(dilate_frame, text="4")
        self.kernel_width_label.pack()

        ttk.Label(dilate_frame, text="Kernel Height:").pack()
        self.kernel_height = tk.IntVar(value=4)
        kernel_h_scale = ttk.Scale(dilate_frame, from_=1, to=20, variable=self.kernel_height, orient=tk.HORIZONTAL)
        kernel_h_scale.pack(fill=tk.X)
        kernel_h_scale.bind("<ButtonRelease-1>", self.on_parameter_change)

        self.kernel_height_label = ttk.Label(dilate_frame, text="4")
        self.kernel_height_label.pack()

        ttk.Label(dilate_frame, text="Iterations:").pack()
        self.dilate_iterations = tk.IntVar(value=1)
        iter_scale = ttk.Scale(dilate_frame, from_=1, to=10, variable=self.dilate_iterations, orient=tk.HORIZONTAL)
        iter_scale.pack(fill=tk.X)
        iter_scale.bind("<ButtonRelease-1>", self.on_parameter_change)

        self.iter_label = ttk.Label(dilate_frame, text="1")
        self.iter_label.pack()

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
        self.notebook.add(processed_frame, text="Detected Regions")

        self.processed_canvas = tk.Canvas(processed_frame, bg="white")
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        self.bind_canvas_events(self.processed_canvas)

        thresh_frame = ttk.Frame(self.notebook)
        self.notebook.add(thresh_frame, text="Threshold")

        self.thresh_canvas = tk.Canvas(thresh_frame, bg="white")
        self.thresh_canvas.pack(fill=tk.BOTH, expand=True)
        self.bind_canvas_events(self.thresh_canvas)

        dilated_frame = ttk.Frame(self.notebook)
        self.notebook.add(dilated_frame, text="Dilated")

        self.dilated_canvas = tk.Canvas(dilated_frame, bg="white")
        self.dilated_canvas.pack(fill=tk.BOTH, expand=True)
        self.bind_canvas_events(self.dilated_canvas)

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
        image = self.original_image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, params["line_removal_threshold"], 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((params["line_removal_kernel_size"][1], params["line_removal_kernel_size"][0]), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=params["line_removal_iterations"])

        self.thresh_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.dilated_image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.processed_image = image.copy()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(self.processed_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def on_parameter_change(self, *args):
        self.process_image()
        self.update_parameters_display()
        self.rebuild_photos()

    def on_tab_changed(self, event):
        current_tab = self.notebook.index(self.notebook.select())
        canvas_photo_pairs = [
            (self.original_canvas, self.original_photo),
            (self.processed_canvas, self.processed_photo),
            (self.thresh_canvas, self.thresh_photo),
            (self.dilated_canvas, self.dilated_photo),
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
        self.thresh_photo = self.create_photo(self.thresh_image)
        self.dilated_photo = self.create_photo(self.dilated_image)
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
        self.draw_photo_on_canvas(self.thresh_photo, self.thresh_canvas)
        self.draw_photo_on_canvas(self.dilated_photo, self.dilated_canvas)

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
            "line_removal_threshold": self.thresh_value.get(),
            "line_removal_kernel_size": (self.kernel_width.get(), self.kernel_height.get()),
            "line_removal_iterations": self.dilate_iterations.get(),
        }

    def update_parameters_display(self):
        params = self.get_current_parameters()

        self.params_text.delete(1.0, tk.END)

        display_text = "=== CURRENT PARAMETERS ===\n\n"
        display_text += f"THRESHOLD:\n"
        display_text += f"  Value: {params['line_removal_threshold']}\n\n"
        display_text += f"DILATION KERNEL:\n"
        display_text += f"  Width: {params['line_removal_kernel_size'][0]}\n"
        display_text += f"  Height: {params['line_removal_kernel_size'][1]}\n\n"
        display_text += f"DILATION ITERATIONS:\n"
        display_text += f"  Value: {params['line_removal_iterations']}\n"

        self.params_text.insert(1.0, display_text)

        self.thresh_value_label.config(text=str(params["line_removal_threshold"]))
        self.kernel_width_label.config(text=str(params["line_removal_kernel_size"][0]))
        self.kernel_height_label.config(text=str(params["line_removal_kernel_size"][1]))
        self.iter_label.config(text=str(params["line_removal_iterations"]))

    def copy_params_to_clipboard(self):
        params = self.get_current_parameters()

        clipboard_content = f"""TEXT_BBOX_CONFIG = {{
    "threshold": {params["line_removal_threshold"]},
    "kernel_size": {params["line_removal_kernel_size"]},
    "iterations": {params["line_removal_iterations"]},
}}"""

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

                self.thresh_value.set(params["line_removal_threshold"])
                kernel_size = params["line_removal_kernel_size"]
                self.kernel_width.set(kernel_size[0])
                self.kernel_height.set(kernel_size[1])
                self.dilate_iterations.set(params["line_removal_iterations"])

                self.on_parameter_change()
                messagebox.showinfo("Success", f"Parameters loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load parameters: {str(e)}")


def main():
    root = tk.Tk()
    app = FindTextBboxes(root)
    root.bind("<ButtonRelease-1>", app.end_pan)
    root.mainloop()


if __name__ == "__main__":
    main()
