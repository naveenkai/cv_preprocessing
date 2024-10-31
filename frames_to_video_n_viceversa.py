import cv2
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from tqdm import tqdm

class VideoConverter:
    def __init__(self, window):
        self.window = window
        self.window.title("Video-Frame Converter")
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.fps = tk.IntVar(value=30)
        self.width = tk.IntVar(value=0)
        self.height = tk.IntVar(value=0)
        self.mode = tk.StringVar(value="frames_to_video")
        
        self.create_gui()
        
    def create_gui(self):
        # Mode selection
        mode_frame = ttk.Frame(self.window)
        mode_frame.pack(padx=10, pady=5, fill=tk.X)
        
        ttk.Radiobutton(mode_frame, text="Frames to Video", 
                       variable=self.mode, value="frames_to_video").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(mode_frame, text="Video to Frames", 
                       variable=self.mode, value="video_to_frames").pack(side=tk.LEFT, padx=5)
        
        # Input selection
        input_frame = ttk.Frame(self.window)
        input_frame.pack(padx=10, pady=5, fill=tk.X)
        
        self.input_label = ttk.Label(input_frame, text="Input Folder:")
        self.input_label.pack(side=tk.LEFT, padx=5)
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(input_frame, text="Browse", command=self.browse_input).pack(side=tk.LEFT)
        
        # Output selection
        output_frame = ttk.Frame(self.window)
        output_frame.pack(padx=10, pady=5, fill=tk.X)
        
        self.output_label = ttk.Label(output_frame, text="Output File:")
        self.output_label.pack(side=tk.LEFT, padx=5)
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side=tk.LEFT)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(self.window, text="Parameters")
        params_frame.pack(padx=10, pady=5, fill=tk.X)
        
        # FPS setting
        fps_frame = ttk.Frame(params_frame)
        fps_frame.pack(padx=5, pady=5, fill=tk.X)
        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(fps_frame, textvariable=self.fps, width=10).pack(side=tk.LEFT, padx=5)
        
        # Resolution settings
        res_frame = ttk.Frame(params_frame)
        res_frame.pack(padx=5, pady=5, fill=tk.X)
        ttk.Label(res_frame, text="Width:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(res_frame, textvariable=self.width, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(res_frame, text="Height:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(res_frame, textvariable=self.height, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(res_frame, text="(0 for original size)").pack(side=tk.LEFT, padx=5)
        
        # Convert button
        ttk.Button(self.window, text="Convert", command=self.convert).pack(pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.window, textvariable=self.status_var)
        self.status_label.pack(pady=5)
        
        # Bind mode change
        self.mode.trace('w', self.update_labels)
        
    def update_labels(self, *args):
        if self.mode.get() == "frames_to_video":
            self.input_label.config(text="Input Folder:")
            self.output_label.config(text="Output File:")
        else:
            self.input_label.config(text="Input Video:")
            self.output_label.config(text="Output Folder:")
            
    def browse_input(self):
        if self.mode.get() == "frames_to_video":
            folder = filedialog.askdirectory(title="Select Input Folder")
            if folder:
                self.input_path.set(folder)
        else:
            file = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
            )
            if file:
                self.input_path.set(file)
            
    def browse_output(self):
        if self.mode.get() == "frames_to_video":
            file = filedialog.asksaveasfilename(
                title="Save Video As",
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
            )
            if file:
                self.output_path.set(file)
        else:
            folder = filedialog.askdirectory(title="Select Output Folder")
            if folder:
                self.output_path.set(folder)
                
    def convert(self):
        if self.mode.get() == "frames_to_video":
            self.frames_to_video()
        else:
            self.video_to_frames()
            
    def frames_to_video(self):
        input_folder = self.input_path.get()
        output_file = self.output_path.get()
        
        if not input_folder or not output_file:
            messagebox.showerror("Error", "Please select both input folder and output file")
            return
            
        try:
            valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
            image_files = [f for f in os.listdir(input_folder) 
                         if f.lower().endswith(valid_extensions)]
            
            if not image_files:
                messagebox.showerror("Error", f"No image files found in {input_folder}")
                return
                
            image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
            first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
            
            size = (self.width.get(), self.height.get()) if self.width.get() > 0 and self.height.get() > 0 else (first_image.shape[1], first_image.shape[0])
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, self.fps.get(), size)
            
            self.status_var.set(f"Converting {len(image_files)} frames...")
            self.window.update()
            
            for image_file in tqdm(image_files):
                frame = cv2.imread(os.path.join(input_folder, image_file))
                if frame is not None:
                    if frame.shape[:2][::-1] != size:
                        frame = cv2.resize(frame, size)
                    out.write(frame)
            
            out.release()
            messagebox.showinfo("Success", f"Video created successfully!\nDuration: {len(image_files)/self.fps.get():.2f} seconds")
            self.status_var.set("Ready")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error occurred")
            
    def video_to_frames(self):
        input_file = self.input_path.get()
        output_folder = self.output_path.get()
        
        if not input_file or not output_folder:
            messagebox.showerror("Error", "Please select both input video and output folder")
            return
            
        try:
            cap = cv2.VideoCapture(input_file)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video file")
                return
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.status_var.set(f"Extracting {total_frames} frames...")
            self.window.update()
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if self.width.get() > 0 and self.height.get() > 0:
                    frame = cv2.resize(frame, (self.width.get(), self.height.get()))
                    
                cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count:06d}.png"), frame)
                frame_count += 1
                
            cap.release()
            messagebox.showinfo("Success", f"Extracted {frame_count} frames successfully!")
            self.status_var.set("Ready")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error occurred")

def main():
    root = tk.Tk()
    app = VideoConverter(root)
    root.mainloop()

if __name__ == "__main__":
    main()