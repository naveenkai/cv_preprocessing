#Test your Yolo models

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import threading
from ultralytics import YOLO
import os
from datetime import datetime
from pathlib import Path
import zipfile
import re
import numpy as np

class YOLOv8Interface:
    def __init__(self, window):
        self.window = window
        self.window.title("YOLOv8 Inference Interface")
        self.window.geometry("1200x800")
        
        # Initialize variables
        self.model = None
        self.video_source = None
        self.is_playing = False
        self.thread = None
        self.model_path = None
        self.video_path = None
        self.image_folder = None
        self.image_files = []
        self.current_image_idx = 0
        self.output_folder = None
        self.saved_frames = []
        self.is_image_mode = False
        
        # Add time format validator
        self.time_pattern = r'^(?:(?:([01]?\d|2[0-3]):)?([0-5]?\d):)?([0-5]?\d)$'
        
        # Add new variables
        self.brightness = tk.DoubleVar(value=1.0)  # Default brightness multiplier
        self.contrast = tk.DoubleVar(value=1.0)    # Default contrast multiplier
        self.iou_threshold = tk.DoubleVar(value=0.45)  # Default IOU threshold
        self.nms_enabled = tk.BooleanVar(value=True)  # Default NMS enabled
        
        # Add hyperparameter display variables
        self.hyper_params_text = None
        self.param_string = tk.StringVar()
        
        # Create GUI elements
        self.create_gui()
        
    def create_gui(self):
        # Create frames
        self.control_frame = ttk.Frame(self.window)
        self.control_frame.pack(pady=10)
        
        self.media_control_frame = ttk.Frame(self.window)
        self.media_control_frame.pack(pady=5)
        
        self.video_frame = ttk.Frame(self.window)
        self.video_frame.pack(pady=10)
        
        self.info_frame = ttk.Frame(self.window)
        self.info_frame.pack(pady=10, fill=tk.X, padx=10)
        
        # Create buttons - Top row
        ttk.Button(self.control_frame, text="Load Model (.pt file)", 
                  command=self.load_model).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.control_frame, text="Load Video", 
                  command=self.load_video).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.control_frame, text="Load Image Folder", 
                  command=self.load_image_folder).pack(side=tk.LEFT, padx=5)
        
        self.play_button = ttk.Button(self.control_frame, text="Start Detection", 
                                    command=self.toggle_detection)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(self.control_frame, text="Stop", 
                                    command=self.stop_detection, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.control_frame, text="Save Output", 
                  command=self.save_output).pack(side=tk.LEFT, padx=5)
        
        # Add model parameter frame
        self.model_params_frame = ttk.LabelFrame(self.window, text="Model Parameters")
        self.model_params_frame.pack(pady=5, padx=10, fill=tk.X)

        # Confidence threshold
        conf_frame = ttk.Frame(self.model_params_frame)
        conf_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT, padx=5)
        self.conf_threshold = tk.DoubleVar(value=0.25)
        self.conf_slider = ttk.Scale(conf_frame, from_=0.0, to=1.0, 
                                   variable=self.conf_threshold, orient=tk.HORIZONTAL)
        self.conf_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(conf_frame, textvariable=tk.StringVar(value=lambda: f"{self.conf_threshold.get():.2f}")).pack(side=tk.LEFT, padx=5)

        # IOU threshold
        iou_frame = ttk.Frame(self.model_params_frame)
        iou_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(iou_frame, text="IOU:").pack(side=tk.LEFT, padx=5)
        self.iou_slider = ttk.Scale(iou_frame, from_=0.0, to=1.0, 
                                  variable=self.iou_threshold, orient=tk.HORIZONTAL)
        self.iou_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(iou_frame, textvariable=tk.StringVar(value=lambda: f"{self.iou_threshold.get():.2f}")).pack(side=tk.LEFT, padx=5)

        # NMS checkbox
        nms_frame = ttk.Frame(self.model_params_frame)
        nms_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(nms_frame, text="NMS:").pack(side=tk.LEFT, padx=5)
        self.nms_checkbox = ttk.Checkbutton(nms_frame, 
                                          text="Enable NMS",
                                          variable=self.nms_enabled)
        self.nms_checkbox.pack(side=tk.LEFT, padx=5)

        # Add image adjustment frame
        self.img_adjust_frame = ttk.LabelFrame(self.window, text="Image Adjustments")
        self.img_adjust_frame.pack(pady=5, padx=10, fill=tk.X)

        # Brightness control
        brightness_frame = ttk.Frame(self.img_adjust_frame)
        brightness_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(brightness_frame, text="Brightness:").pack(side=tk.LEFT, padx=5)
        self.brightness_slider = ttk.Scale(brightness_frame, from_=0.1, to=3.0, 
                                         variable=self.brightness, orient=tk.HORIZONTAL)
        self.brightness_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(brightness_frame, textvariable=tk.StringVar(value=lambda: f"{self.brightness.get():.2f}")).pack(side=tk.LEFT, padx=5)

        # Contrast control
        contrast_frame = ttk.Frame(self.img_adjust_frame)
        contrast_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(contrast_frame, text="Contrast:").pack(side=tk.LEFT, padx=5)
        self.contrast_slider = ttk.Scale(contrast_frame, from_=0.1, to=3.0, 
                                       variable=self.contrast, orient=tk.HORIZONTAL)
        self.contrast_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(contrast_frame, textvariable=tk.StringVar(value=lambda: f"{self.contrast.get():.2f}")).pack(side=tk.LEFT, padx=5)

        # Reset adjustments button
        ttk.Button(self.img_adjust_frame, text="Reset Adjustments", 
                  command=self.reset_adjustments).pack(pady=5)
        
        # Create image navigation controls
        self.prev_button = ttk.Button(self.media_control_frame, text="Previous", 
                                    command=self.prev_image, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(self.media_control_frame, text="Next", 
                                    command=self.next_image, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        self.image_counter = ttk.Label(self.media_control_frame, text="Image: 0/0")
        self.image_counter.pack(side=tk.LEFT, padx=5)
        
        # Create video label
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack()
        
        # Create info text
        self.info_text = tk.Text(self.info_frame, height=10, width=100)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(self.info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        # Replace time range frame code
        self.time_range_frame = ttk.Frame(self.window)
        self.time_range_frame.pack(pady=5)
        
        ttk.Label(self.time_range_frame, text="Start Time (HH:MM:SS):").pack(side=tk.LEFT, padx=5)
        self.start_time_var = tk.StringVar(value="00:00:00")
        self.start_time_entry = ttk.Entry(self.time_range_frame, textvariable=self.start_time_var, width=10)
        self.start_time_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.time_range_frame, text="End Time (HH:MM:SS):").pack(side=tk.LEFT, padx=5)
        self.end_time_var = tk.StringVar()
        self.end_time_entry = ttk.Entry(self.time_range_frame, textvariable=self.end_time_var, width=10)
        self.end_time_entry.pack(side=tk.LEFT, padx=5)
        
        # Add hyperparameter display frame
        self.hyper_params_frame = ttk.LabelFrame(self.window, text="Current Hyperparameters")
        self.hyper_params_frame.pack(pady=5, padx=10, fill=tk.X)
        
        self.hyper_params_text = ttk.Label(self.hyper_params_frame, 
                                         textvariable=self.param_string,
                                         justify=tk.LEFT)
        self.hyper_params_text.pack(pady=5, padx=5)

        # Add trace to all parameters
        self.conf_threshold.trace_add("write", self.update_param_display)
        self.iou_threshold.trace_add("write", self.update_param_display)
        self.nms_enabled.trace_add("write", self.update_param_display)
        self.brightness.trace_add("write", self.update_param_display)
        self.contrast.trace_add("write", self.update_param_display)
    
    def update_param_display(self, *args):
        """Update the hyperparameter display"""
        params = (
            f"Confidence Threshold: {self.conf_threshold.get():.3f}\n"
            f"IOU Threshold: {self.iou_threshold.get():.3f}\n"
            f"NMS: {'Enabled' if self.nms_enabled.get() else 'Disabled'}\n"
            f"Brightness: {self.brightness.get():.2f}x\n"
            f"Contrast: {self.contrast.get():.2f}x"
        )
        self.param_string.set(params)
        
        # Log changes to info panel
        self.log_info("Parameters updated:")
        self.log_info(params)
    
    def time_to_seconds(self, time_str):
        """Convert HH:MM:SS to seconds"""
        if not re.match(self.time_pattern, time_str):
            raise ValueError(f"Invalid time format: {time_str}. Use HH:MM:SS format")
            
        parts = time_str.split(':')
        parts = ['00'] * (3 - len(parts)) + parts  # Pad with zeros if hours/minutes missing
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    
    def seconds_to_time(self, seconds):
        """Convert seconds to HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def load_model(self):
        self.model_path = filedialog.askopenfilename(filetypes=[("PT files", "*.pt")])
        if self.model_path:
            try:
                self.model = YOLO(self.model_path)
                self.log_info(f"Model loaded successfully: {os.path.basename(self.model_path)}")
            except Exception as e:
                self.log_info(f"Error loading model: {str(e)}")
                self.model = None
    
    def load_video(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if self.video_path:
            self.is_image_mode = False
            self.image_files = []
            
            # Get video duration
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_duration = total_frames / fps
            cap.release()
            
            # Update end time entry with video duration in HH:MM:SS
            self.end_time_var.set(self.seconds_to_time(self.video_duration))
            self.log_info(f"Video loaded: {os.path.basename(self.video_path)} "
                         f"(Duration: {self.seconds_to_time(self.video_duration)})")
            self.update_navigation_buttons()
            
            # Update parameter display
            self.update_param_display()
            self.log_info("Initial parameters set")
    
    def load_image_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_folder = folder_path
            self.image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            self.image_files.sort()
            if self.image_files:
                self.is_image_mode = True
                self.current_image_idx = 0
                self.video_path = None
                self.log_info(f"Loaded {len(self.image_files)} images from folder")
                self.update_navigation_buttons()
                self.process_current_image()
            else:
                self.log_info("No images found in selected folder")
    
    def update_navigation_buttons(self):
        if self.is_image_mode and self.image_files:
            self.prev_button.configure(state=tk.NORMAL)
            self.next_button.configure(state=tk.NORMAL)
            self.update_image_counter()
        else:
            self.prev_button.configure(state=tk.DISABLED)
            self.next_button.configure(state=tk.DISABLED)
            self.image_counter.configure(text="Image: 0/0")
    
    def update_image_counter(self):
        if self.image_files:
            self.image_counter.configure(
                text=f"Image: {self.current_image_idx + 1}/{len(self.image_files)}"
            )
    
    def next_image(self):
        if self.current_image_idx < len(self.image_files) - 1:
            self.current_image_idx += 1
            self.process_current_image()
            self.update_image_counter()
    
    def prev_image(self):
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.process_current_image()
            self.update_image_counter()
    
    def process_current_image(self):
        if not self.image_files or not self.model:
            return
            
        image_path = os.path.join(self.image_folder, self.image_files[self.current_image_idx])
        frame = cv2.imread(image_path)
        if frame is None:
            self.log_info(f"Error loading image: {image_path}")
            return
            
        # Process frame with current settings
        annotated_frame = self.process_frame(frame)
        
        # Display the frame
        self.display_frame(annotated_frame)
        
        # Add to saved frames if we're saving
        if self.output_folder:
            output_path = os.path.join(
                self.output_folder, 
                f"pred_{os.path.basename(image_path)}"
            )
            cv2.imwrite(output_path, annotated_frame)
    
    def draw_predictions(self, frame, results):
        annotated_frame = frame.copy()
        boxes = results.boxes.cpu().numpy()
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence and class
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            # Draw box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Log detection
            self.log_info(f"Detected {class_name} with confidence {confidence:.2f}")
        
        return annotated_frame
    
    def display_frame(self, frame):
        # Convert to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to fit display (maintaining aspect ratio)
        height, width = rgb_frame.shape[:2]
        max_size = (800, 600)
        scale = min(max_size[0]/width, max_size[1]/height)
        new_size = (int(width*scale), int(height*scale))
        rgb_frame = cv2.resize(rgb_frame, new_size)
        
        # Convert to PhotoImage
        image = Image.fromarray(rgb_frame)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update video label
        self.video_label.configure(image=photo)
        self.video_label.image = photo
    
    def toggle_detection(self):
        if not self.is_playing:
            if self.model is None:
                self.log_info("Please load a model first!")
                return
            if not (self.video_path or self.image_files):
                self.log_info("Please load a video or image folder first!")
                return
                
            self.is_playing = True
            self.play_button.configure(text="Pause Detection")
            self.stop_button.configure(state=tk.NORMAL)
            
            if not self.is_image_mode:
                self.thread = threading.Thread(target=self.run_video_detection)
                self.thread.daemon = True
                self.thread.start()
        else:
            self.is_playing = False
            self.play_button.configure(text="Resume Detection")
    
    def stop_detection(self):
        self.is_playing = False
        self.play_button.configure(text="Start Detection")
        self.stop_button.configure(state=tk.DISABLED)
        if self.output_folder:
            self.save_output()
    
    def run_video_detection(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        # Get time range
        try:
            start_time = self.time_to_seconds(self.start_time_var.get())
            end_time = (self.time_to_seconds(self.end_time_var.get()) 
                       if self.end_time_var.get() 
                       else self.video_duration)
            
            if start_time < 0 or end_time > self.video_duration or start_time >= end_time:
                self.log_info("Invalid time range specified")
                cap.release()
                self.stop_detection()
                return
                
            # Set starting frame position
            start_frame = int(start_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Calculate end frame
            end_frame = int(end_time * fps)
            
            self.log_info(f"Processing video from {self.seconds_to_time(start_time)} "
                         f"to {self.seconds_to_time(end_time)}")
            
        except ValueError as e:
            self.log_info(f"Time format error: {str(e)}")
            cap.release()
            self.stop_detection()
            return
        
        while self.is_playing:
            ret, frame = cap.read()
            if not ret or frame_count >= (end_frame - start_frame):
                self.stop_detection()
                break
            
            # Process frame with current settings
            annotated_frame = self.process_frame(frame)
            
            # Display the frame
            self.display_frame(annotated_frame)
            
            # Save frame if output folder is set
            if self.output_folder:
                output_path = os.path.join(
                    self.output_folder, 
                    f"frame_{frame_count:06d}.jpg"
                )
                cv2.imwrite(output_path, annotated_frame)
            
            frame_count += 1
            self.window.update()
            
        cap.release()
    
    def save_output(self):
        if not self.output_folder:
            self.output_folder = filedialog.askdirectory(title="Select Output Folder")
            if not self.output_folder:
                return
            
            temp_folder = os.path.join(self.output_folder, "temp_frames")
            os.makedirs(temp_folder, exist_ok=True)
            self.log_info(f"Processing outputs to: {self.output_folder}")
            
            try:
                if self.is_image_mode:
                    # Process all images in folder
                    for idx, image_file in enumerate(self.image_files):
                        self.current_image_idx = idx
                        image_path = os.path.join(self.image_folder, image_file)
                        frame = cv2.imread(image_path)
                        results = self.model(frame, conf=self.conf_threshold.get())[0]
                        annotated_frame = self.draw_predictions(frame, results)
                        output_path = os.path.join(temp_folder, f"pred_{image_file}")
                        cv2.imwrite(output_path, annotated_frame)
                else:
                    # Process video frames
                    cap = cv2.VideoCapture(self.video_path)
                    frame_count = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        results = self.model(frame, conf=self.conf_threshold.get())[0]
                        annotated_frame = self.draw_predictions(frame, results)
                        output_path = os.path.join(temp_folder, f"frame_{frame_count:06d}.jpg")
                        cv2.imwrite(output_path, annotated_frame)
                        frame_count += 1
                    cap.release()
                
                # Create zip file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_filename = os.path.join(self.output_folder, f"predictions_{timestamp}.zip")
                with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(temp_folder):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, temp_folder)
                            zipf.write(file_path, arcname)
                
                self.log_info(f"Successfully saved predictions to: {zip_filename}")
                
            finally:
                # Clean up temporary folder
                import shutil
                if os.path.exists(temp_folder):
                    shutil.rmtree(temp_folder)
            
            self.output_folder = None  # Reset output folder for next save
    
    def log_info(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.info_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.info_text.see(tk.END)
    
    def reset_adjustments(self):
        """Reset image adjustments to default values"""
        self.brightness.set(1.0)
        self.contrast.set(1.0)
        self.conf_threshold.set(0.25)
        self.iou_threshold.set(0.45)
        self.nms_enabled.set(True)
        
        # Update parameter display
        self.update_param_display()
        self.log_info("Parameters reset to defaults")
    
    def adjust_image(self, frame):
        """Apply brightness and contrast adjustments to the frame"""
        # Convert to float for calculations
        adjusted = frame.astype(float)
        
        # Apply brightness
        adjusted *= self.brightness.get()
        
        # Apply contrast
        adjusted = (adjusted - 128) * self.contrast.get() + 128
        
        # Clip values to valid range
        adjusted = np.clip(adjusted, 0, 255)
        
        return adjusted.astype(np.uint8)

    def process_frame(self, frame):
        """Process a single frame with current settings"""
        # Apply image adjustments
        adjusted_frame = self.adjust_image(frame)
        
        # Run inference with current parameters
        results = self.model(
            adjusted_frame,
            conf=self.conf_threshold.get(),
            iou=self.iou_threshold.get(),
            nms=self.nms_enabled.get()  # Now passing boolean value
        )[0]
        
        return self.draw_predictions(adjusted_frame, results)

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv8Interface(root)
    root.mainloop()
