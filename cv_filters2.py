import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional
from functools import partial
from scipy import ndimage  # Add this import

class ImagePreprocessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Image Preprocessing Viewer")
        self.root.geometry("1200x800")

        self.original_image: Optional[np.ndarray] = None
        self.current_image: Optional[np.ndarray] = None
        self.photo = None
        self.last_filter = None

        self.setup_ui()
        self.create_filter_functions()

        # Bind window resize event
        self.root.bind("<Configure>", self.delayed_resize)
        self.resize_timer = None

    def setup_ui(self):
        # Main container
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left panel for controls (reduced width)
        left_panel = ttk.Frame(main_container, width=300)
        main_container.add(left_panel, weight=1)

        # Right panel for image display (increased width)
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=4)

        # Create a canvas with scrollbar for the left panel
        canvas = tk.Canvas(left_panel, width=280)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Upload button
        upload_btn = ttk.Button(scrollable_frame, text="Upload Image", command=self.load_image)
        upload_btn.pack(fill=tk.X, padx=5, pady=5)

        # Reset button
        reset_btn = ttk.Button(scrollable_frame, text="Reset Image", command=self.reset_image)
        reset_btn.pack(fill=tk.X, padx=5, pady=5)

        # Save button
        save_btn = ttk.Button(scrollable_frame, text="Save Image", command=self.save_image)
        save_btn.pack(fill=tk.X, padx=5, pady=5)

        # Filter categories and options
        self.setup_filter_categories(scrollable_frame)

        # Parameter controls
        self.setup_parameter_controls(scrollable_frame)

        # Image display
        self.image_label = ttk.Label(right_panel)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_filter_functions(self):
        self.filter_functions = {
            "Grayscale": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img,
            "BGR2RGB": lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            "Increase Brightness": lambda img: cv2.convertScaleAbs(img, alpha=1.2, beta=30),
            "Increase Contrast": lambda img: cv2.convertScaleAbs(img, alpha=1.4, beta=0),
            "Equalize Histogram": self.equalize_histogram,
            "Gaussian Blur": lambda img: cv2.GaussianBlur(img, (int(self.kernel_var.get()), int(self.kernel_var.get())), self.sigma_var.get()),
            "Median Blur": lambda img: cv2.medianBlur(img, int(self.kernel_var.get())),
            "Bilateral Filter": lambda img: cv2.bilateralFilter(img, 9, 75, 75),
            "Box Blur": lambda img: cv2.blur(img, (int(self.kernel_var.get()), int(self.kernel_var.get()))),
            "Sharpen": self.sharpen_image,
            "Sobel X": self.sobel_x,
            "Sobel Y": self.sobel_y,
            "Laplacian": self.laplacian,
            "Canny Edge": self.canny_edge,
            "Prewitt": self.prewitt,
            "Binary Threshold": self.binary_threshold,
            "Adaptive Threshold": self.adaptive_threshold,
            "Otsu Threshold": self.otsu_threshold,
            "Triangle Threshold": self.triangle_threshold,
            "Erosion": self.erosion,
            "Dilation": self.dilation,
            "Opening": self.opening,
            "Closing": self.closing,
            "Gradient": self.morphological_gradient,
            "Wrinkle Detection": self.detect_wrinkles,
            "Shrinkage Detection": self.detect_shrinkage,
            "Surface Defect Map": self.surface_defect_map,
            "Edge Defect Detection": self.edge_defect_detection,
            "Texture Analysis": self.texture_analysis,
            "Pattern Deviation": self.pattern_deviation,
            "Local Binary Pattern": self.local_binary_pattern,
            "Pseudo Color Heat Map": self.pseudo_color_heat_map,
            "Temperature Gradient": self.temperature_gradient,
            "Hot Spot Detection": self.hot_spot_detection,
            "Cold Spot Detection": self.cold_spot_detection,
            "Thermal Edge Detection": self.thermal_edge_detection,
            "Temperature Range Filter": self.temperature_range_filter,
            "Thermal Contrast Enhance": self.thermal_contrast_enhance
        }

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.current_image = self.original_image.copy()
            self.last_filter = None
            self.update_image_display()

    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.last_filter = None
            self.update_image_display()

    def apply_filter(self, filter_name):
        if self.current_image is not None and filter_name in self.filter_functions:
            try:
                self.current_image = self.filter_functions[filter_name](self.current_image)
                self.last_filter = filter_name
                self.update_image_display()
            except Exception as e:
                messagebox.showerror("Error", f"Error applying filter: {str(e)}")

    def update_image_display(self):
        if self.current_image is not None:
            # Convert to RGB for display only if necessary
            if len(self.current_image.shape) == 2:
                display_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB)
            elif self.current_image.shape[2] == 3:
                display_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            else:
                display_image = self.current_image

            # Get the size of the image display area
            display_width = self.image_label.winfo_width()
            display_height = self.image_label.winfo_height()

            # If the label size is not yet updated, use a default size
            if display_width <= 1 or display_height <= 1:
                display_width, display_height = 800, 600

            # Calculate the scaling factor to fit the image while maintaining aspect ratio
            img_height, img_width = display_image.shape[:2]
            scale = min(display_width/img_width, display_height/img_height)

            # Calculate new dimensions
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)

            # Resize the image using PIL for better quality
            pil_image = Image.fromarray(display_image)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(pil_image)
            self.image_label.configure(image=self.photo)

    def delayed_resize(self, event):
        if self.resize_timer is not None:
            self.root.after_cancel(self.resize_timer)
        self.resize_timer = self.root.after(200, self.update_image_display)

    def setup_filter_categories(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Color Adjustments
        color_frame = ttk.Frame(notebook)
        notebook.add(color_frame, text="Color")
        color_filters = [
            "Grayscale", "BGR2RGB", "Increase Brightness",
            "Increase Contrast", "Equalize Histogram"
        ]
        self.create_filter_buttons(color_frame, color_filters)

        # Blur/Sharpen
        blur_frame = ttk.Frame(notebook)
        notebook.add(blur_frame, text="Blur/Sharpen")
        blur_filters = [
            "Gaussian Blur", "Median Blur", "Bilateral Filter",
            "Box Blur", "Sharpen"
        ]
        self.create_filter_buttons(blur_frame, blur_filters)

        # Edge Detection
        edge_frame = ttk.Frame(notebook)
        notebook.add(edge_frame, text="Edges")
        edge_filters = [
            "Sobel X", "Sobel Y", "Laplacian",
            "Canny Edge", "Prewitt"
        ]
        self.create_filter_buttons(edge_frame, edge_filters)

        # Thresholding
        threshold_frame = ttk.Frame(notebook)
        notebook.add(threshold_frame, text="Threshold")
        threshold_filters = [
            "Binary Threshold", "Adaptive Threshold",
            "Otsu Threshold", "Triangle Threshold"
        ]
        self.create_filter_buttons(threshold_frame, threshold_filters)

        # Morphological
        morph_frame = ttk.Frame(notebook)
        notebook.add(morph_frame, text="Morphological")
        morph_filters = [
            "Erosion", "Dilation", "Opening",
            "Closing", "Gradient"
        ]
        self.create_filter_buttons(morph_frame, morph_filters)

        # Defect Detection
        defect_frame = ttk.Frame(notebook)
        notebook.add(defect_frame, text="Defect Detection")
        defect_filters = [
            "Wrinkle Detection", "Shrinkage Detection",
            "Surface Defect Map", "Edge Defect Detection",
            "Texture Analysis", "Pattern Deviation",
            "Local Binary Pattern"
        ]
        self.create_filter_buttons(defect_frame, defect_filters)

        # Thermal Analysis
        thermal_frame = ttk.Frame(notebook)
        notebook.add(thermal_frame, text="Thermal Analysis")
        thermal_filters = [
            "Pseudo Color Heat Map", "Temperature Gradient",
            "Hot Spot Detection", "Cold Spot Detection",
            "Thermal Edge Detection", "Temperature Range Filter",
            "Thermal Contrast Enhance"
        ]
        self.create_filter_buttons(thermal_frame, thermal_filters)

    def create_filter_buttons(self, parent, filters):
        for filter_name in filters:
            btn = ttk.Button(
                parent,
                text=filter_name,
                command=lambda f=filter_name: self.apply_filter(f)
            )
            btn.pack(fill=tk.X, padx=5, pady=2)

    def setup_parameter_controls(self, parent):
        param_frame = ttk.LabelFrame(parent, text="Parameters")
        param_frame.pack(fill=tk.X, padx=5, pady=5)

        # Kernel size
        ttk.Label(param_frame, text="Kernel Size:").pack(anchor=tk.W, padx=5)
        self.kernel_var = tk.StringVar(value="3")
        kernel_combo = ttk.Combobox(
            param_frame,
            textvariable=self.kernel_var,
            values=["3", "5", "7", "9", "11"]
        )
        kernel_combo.pack(fill=tk.X, padx=5, pady=2)

        # Threshold value
        ttk.Label(param_frame, text="Threshold:").pack(anchor=tk.W, padx=5)
        self.threshold_var = tk.IntVar(value=127)
        threshold_scale = ttk.Scale(
            param_frame,
            from_=0,
            to=255,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL
        )
        threshold_scale.pack(fill=tk.X, padx=5, pady=2)

        # Sigma value for Gaussian
        ttk.Label(param_frame, text="Sigma:").pack(anchor=tk.W, padx=5)
        self.sigma_var = tk.DoubleVar(value=1.0)
        sigma_scale = ttk.Scale(
            param_frame,
            from_=0.1,
            to=5.0,
            variable=self.sigma_var,
            orient=tk.HORIZONTAL
        )
        sigma_scale.pack(fill=tk.X, padx=5, pady=2)

        # Sensitivity for defect detection
        ttk.Label(param_frame, text="Defect Sensitivity:").pack(anchor=tk.W, padx=5)
        self.sensitivity_var = tk.DoubleVar(value=1.0)
        sensitivity_scale = ttk.Scale(
            param_frame,
            from_=0.1,
            to=3.0,
            variable=self.sensitivity_var,
            orient=tk.HORIZONTAL
        )
        sensitivity_scale.pack(fill=tk.X, padx=5, pady=2)

        # Temperature range for thermal analysis
        ttk.Label(param_frame, text="Temperature Range:").pack(anchor=tk.W, padx=5)
        self.temp_min_var = tk.IntVar(value=0)
        self.temp_max_var = tk.IntVar(value=255)
        
        temp_frame = ttk.Frame(param_frame)
        temp_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(temp_frame, text="Min:").pack(side=tk.LEFT)
        ttk.Scale(
            temp_frame,
            from_=0,
            to=255,
            variable=self.temp_min_var,
            orient=tk.HORIZONTAL
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(temp_frame, text="Max:").pack(side=tk.LEFT)
        ttk.Scale(
            temp_frame,
            from_=0,
            to=255,
            variable=self.temp_max_var,
            orient=tk.HORIZONTAL
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def save_image(self):
        if self.current_image is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                cv2.imwrite(file_path, self.current_image)
                messagebox.showinfo("Success", "Image saved successfully!")

    # Filter implementation methods
    def equalize_histogram(self, image):
        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return cv2.equalizeHist(image)

    def sharpen_image(self, image):
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        return cv2.filter2D(image, -1, kernel)

    def sobel_x(self, image):
        gray = self.filter_functions["Grayscale"](image)
        return cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=int(self.kernel_var.get()))

    def sobel_y(self, image):
        gray = self.filter_functions["Grayscale"](image)
        return cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=int(self.kernel_var.get()))

    def laplacian(self, image):
        gray = self.filter_functions["Grayscale"](image)
        return cv2.Laplacian(gray, cv2.CV_64F)

    def canny_edge(self, image):
        gray = self.filter_functions["Grayscale"](image)
        return cv2.Canny(gray, 100, 200)

    def prewitt(self, image):
        gray = self.filter_functions["Grayscale"](image)
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        x = cv2.filter2D(gray, -1, kernelx)
        y = cv2.filter2D(gray, -1, kernely)
        return cv2.addWeighted(x, 0.5, y, 0.5, 0)

    def binary_threshold(self, image):
        gray = self.filter_functions["Grayscale"](image)
        _, binary = cv2.threshold(gray, self.threshold_var.get(), 255, cv2.THRESH_BINARY)
        return binary

    def adaptive_threshold(self, image):
        gray = self.filter_functions["Grayscale"](image)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    def otsu_threshold(self, image):
        gray = self.filter_functions["Grayscale"](image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def triangle_threshold(self, image):
        gray = self.filter_functions["Grayscale"](image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        return binary

    def erosion(self, image):
        kernel = np.ones((int(self.kernel_var.get()), int(self.kernel_var.get())), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    def dilation(self, image):
        kernel = np.ones((int(self.kernel_var.get()), int(self.kernel_var.get())), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    def opening(self, image):
        kernel = np.ones((int(self.kernel_var.get()), int(self.kernel_var.get())), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    def closing(self, image):
        kernel = np.ones((int(self.kernel_var.get()), int(self.kernel_var.get())), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def morphological_gradient(self, image):
        kernel = np.ones((int(self.kernel_var.get()), int(self.kernel_var.get())), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

    def cold_spot_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Apply threshold to identify cold spots
        threshold = self.temp_min_var.get()
        _, cold_spots = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Create result image
        result = image.copy() if len(image.shape) > 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        result[cold_spots > 0] = [255, 0, 0]  # Mark cold spots in blue
        
        return result

    def thermal_edge_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Apply bilateral filter to reduce noise while preserving edges
        smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detect edges using Canny
        min_temp = self.temp_min_var.get()
        max_temp = self.temp_max_var.get()
        edges = cv2.Canny(smoothed, min_temp, max_temp)
        
        # Create color overlay
        result = image.copy() if len(image.shape) > 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        result[edges > 0] = [0, 255, 255]  # Mark thermal edges in yellow
        
        return result

    def temperature_range_filter(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Create mask for temperature range
        min_temp = self.temp_min_var.get()
        max_temp = self.temp_max_var.get()
        
        mask = cv2.inRange(gray, min_temp, max_temp)
        
        # Apply colormap to the masked region
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        result = colored.copy()
        result[mask == 0] = [0, 0, 0]  # Set regions outside range to black
        
        return result

    def thermal_contrast_enhance(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply thermal colormap
        result = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
        
        return result

    def pattern_deviation(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Calculate local standard deviation
        kernel_size = int(self.kernel_var.get())
        local_std = ndimage.generic_filter(gray, np.std, size=kernel_size)
        
        # Normalize deviation map
        deviation_map = cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap
        result = cv2.applyColorMap(deviation_map, cv2.COLORMAP_VIRIDIS)
        
        return result

    def texture_analysis(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Calculate GLCM properties
        glcm = self.graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        contrast = self.graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = self.graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = self.graycoprops(glcm, 'homogeneity')[0, 0]
        energy = self.graycoprops(glcm, 'energy')[0, 0]
        correlation = self.graycoprops(glcm, 'correlation')[0, 0]
        
        # Create a result image with texture properties
        result = np.zeros_like(image)
        cv2.putText(result, f"Contrast: {contrast:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f"Dissimilarity: {dissimilarity:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f"Homogeneity: {homogeneity:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f"Energy: {energy:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f"Correlation: {correlation:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result

    def local_binary_pattern(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        radius = 3
        n_points = 8 * radius
        lbp = self.calculate_lbp(gray, n_points, radius)
        
        # Normalize LBP image
        lbp_normalized = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap
        result = cv2.applyColorMap(lbp_normalized, cv2.COLORMAP_RAINBOW)
        
        return result

    def pseudo_color_heat_map(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return heatmap

    def temperature_gradient(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Calculate gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and convert to uint8
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply colormap
        result = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
        
        return result

    def hot_spot_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Apply threshold to identify hot spots
        threshold = self.temp_max_var.get()
        _, hot_spots = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Create result image
        result = image.copy() if len(image.shape) > 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        result[hot_spots > 0] = [0, 0, 255]  # Mark hot spots in red
        
        return result

    def cold_spot_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Apply threshold to identify cold spots
        threshold = self.temp_min_var.get()
        _, cold_spots = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Create result image
        result = image.copy() if len(image.shape) > 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        result[cold_spots > 0] = [255, 0, 0]  # Mark cold spots in blue
        
        return result

    def thermal_edge_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Apply bilateral filter to reduce noise while preserving edges
        smoothed = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Detect edges using Canny
        min_temp = self.temp_min_var.get()
        max_temp = self.temp_max_var.get()
        edges = cv2.Canny(smoothed, min_temp, max_temp)
        
        # Create color overlay
        result = image.copy() if len(image.shape) > 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        result[edges > 0] = [0, 255, 255]  # Mark thermal edges in yellow
        
        return result

    def temperature_range_filter(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Create mask for temperature range
        min_temp = self.temp_min_var.get()
        max_temp = self.temp_max_var.get()
        
        mask = cv2.inRange(gray, min_temp, max_temp)
        
        # Apply colormap to the masked region
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        result = colored.copy()
        result[mask == 0] = [0, 0, 0]  # Set regions outside range to black
        
        return result

    def thermal_contrast_enhance(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply thermal colormap
        result = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
        
        return result

    # Helper methods
    def graycomatrix(self, image, distances, angles, levels=256):
        glcm = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.uint32)
        for i, d in enumerate(distances):
            for j, a in enumerate(angles):
                dx = int(round(d * np.cos(a)))
                dy = int(round(d * np.sin(a)))
                rows, cols = image.shape
                for x in range(rows):
                    for y in range(cols):
                        if 0 <= x + dx < rows and 0 <= y + dy < cols:
                            i = image[x, y]
                            j = image[x + dx, y + dy]
                            glcm[i, j, i, j] += 1
        return glcm

    def graycoprops(self, P, prop='contrast'):
        (num_level, num_level2, num_dist, num_angle) = P.shape
        if num_level != num_level2:
            raise ValueError('num_level and num_level2 must be equal.')
        I, J = np.ogrid[0:num_level, 0:num_level]
        if prop == 'contrast':
            weights = (I - J) ** 2
            return np.sum(P * weights[..., np.newaxis, np.newaxis], axis=(0, 1))
        elif prop == 'dissimilarity':
            weights = np.abs(I - J)
            return np.sum(P * weights[..., np.newaxis, np.newaxis], axis=(0, 1))
        elif prop == 'homogeneity':
            weights = 1. / (1. + (I - J) ** 2)
            return np.sum(P * weights[..., np.newaxis, np.newaxis], axis=(0, 1))
        elif prop == 'energy':
            return np.sum(P ** 2, axis=(0, 1))
        elif prop == 'correlation':
            mean_i = np.mean(I * np.sum(P, axis=1))
            mean_j = np.mean(J * np.sum(P, axis=0))
            std_i = np.sqrt(np.mean(((I - mean_i) ** 2) * np.sum(P, axis=1)))
            std_j = np.sqrt(np.mean(((J - mean_j) ** 2) * np.sum(P, axis=0)))
            cov = np.mean(((I - mean_i) * (J - mean_j)) * P)
            return cov / (std_i * std_j)
        else:
            raise ValueError('Unknown property')

    def calculate_lbp(self, image, n_points, radius):
        rows, cols = image.shape
        result = np.zeros((rows, cols), dtype=np.uint8)
        for y in range(radius, rows - radius):
            for x in range(radius, cols - radius):
                center = image[y, x]
                code = 0
                for p in range(n_points):
                    r = y - radius * np.sin(2 * np.pi * p / n_points)
                    c = x + radius * np.cos(2 * np.pi * p / n_points)
                    if image[int(r), int(c)] >= center:
                        code |= (1 << p)
                result[y, x] = code
        return result

    def surface_defect_map(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Apply various edge detection methods
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Combine different edge detection results
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        combined_edges = cv2.addWeighted(edge_magnitude, 0.5, np.abs(laplacian), 0.5, 0)
        
        # Normalize and apply threshold
        combined_edges = cv2.normalize(combined_edges, None, 0, 255, cv2.NORM_MINMAX)
        threshold = self.sensitivity_var.get() * 40
        _, defect_map = cv2.threshold(combined_edges, threshold, 255, cv2.THRESH_BINARY)
        
        # Create color-coded defect map
        heat_map = cv2.applyColorMap(defect_map.astype(np.uint8), cv2.COLORMAP_JET)
        
        return heat_map   
    
    def edge_defect_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate the edges to connect nearby edge pixels
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours in the dilated edge image
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a copy of the original image to draw results on
        result = image.copy() if len(image.shape) > 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Analyze each contour
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filter out very small contours
            if area < 50:
                continue
            
            # Calculate shape complexity (a measure of how "rough" the edge is)
            complexity = perimeter * perimeter / (4 * np.pi * area) if area > 0 else 0
            
            # Highlight potential edge defects
            if complexity > 1.5 * self.sensitivity_var.get():
                cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)
        
        return result

    def detect_wrinkles(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate gradients using Sobel
        gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalize and enhance contrast
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply threshold based on sensitivity
        threshold = int(self.sensitivity_var.get() * 50)
        _, wrinkles = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)
        
        # Calculate wrinkle percentage
        total_pixels = wrinkles.size
        wrinkle_pixels = np.count_nonzero(wrinkles)
        wrinkle_percentage = (wrinkle_pixels / total_pixels) * 100
        
        # Create color overlay
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result[wrinkles > 0] = [0, 0, 255]  # Mark wrinkles in red
        
        # Add text with wrinkle percentage
        cv2.putText(result, f"Wrinkle Damage: {wrinkle_percentage:.2f}%", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return result

    def detect_shrinkage(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to create a binary image
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a copy of the original image to draw results on
        result = image.copy() if len(image.shape) > 2 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Create a mask for shrinkage areas
        shrinkage_mask = np.zeros_like(gray)
        
        # Analyze each contour
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Filter out very small contours
            if area < 100:
                continue
            
            # Calculate circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Highlight potential shrinkage areas
            if circularity < 0.5 * self.sensitivity_var.get():
                cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                cv2.drawContours(shrinkage_mask, [contour], -1, 255, -1)
        
        # Calculate shrinkage percentage
        total_pixels = gray.size
        shrinkage_pixels = np.count_nonzero(shrinkage_mask)
        shrinkage_percentage = (shrinkage_pixels / total_pixels) * 100
        
        # Add text with shrinkage percentage
        cv2.putText(result, f"Shrinkage: {shrinkage_percentage:.2f}%", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return result

def main():
    root = tk.Tk()
    app = ImagePreprocessingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

