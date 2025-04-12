import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import os
import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class KMeansClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K-means Clustering Tool v2")
        self.root.geometry("1300x800")  # Increased width to accommodate the right panel
        
        # Define canvas dimensions
        self.canvas_width = 800
        self.canvas_height = 600
        
        # Initialize variables
        self.points = []
        self.img = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255  # White background
        self.k = 3  # Default number of clusters
        self.circle_diameter = 50  # Default diameter for random points
        self.diameter_random = 120  # Random variation in diameter
        self.points_per_click = 10  # Default number of points per click
        self.show_circles = False  # Show circles when generating points
        self.labels = None
        self.centers = None
        
        # Create the points directory if it doesn't exist
        self.points_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "points")
        if not os.path.exists(self.points_dir):
            os.makedirs(self.points_dir)
        
        # Main application frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create three frames: left for controls, center for canvas, right for connection options
        self.left_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Center frame for the drawing canvas
        self.center_frame = ttk.Frame(self.main_frame)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right frame for connection options
        self.right_frame = ttk.LabelFrame(self.main_frame, text="Connection Options")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5, ipadx=10, ipady=10)
        self.right_frame.configure(width=250)  # Fixed width for the connection options panel
        
        # Create the parameter controls
        self.create_control_panel()
        
        # Create the canvas for drawing
        self.create_drawing_canvas()
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Click on canvas to generate points.")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_k_display(self, event=None):
        """Update the display label for k value"""
        self.k_display.config(text=str(int(float(self.k_var.get()))))
        
    def update_diameter_display(self, event=None):
        """Update the display label for circle diameter"""
        self.diameter_display.config(text=str(int(float(self.diameter_var.get()))))
        
    def update_diameter_random_display(self, event=None):
        """Update the display label for random diameter variation"""
        self.diameter_random_display.config(text=str(int(float(self.diameter_random_var.get()))))
        
    def update_points_display(self, event=None):
        """Update the display label for points per click"""
        self.points_display.config(text=str(int(float(self.points_var.get()))))

    def create_control_panel(self):
        """Create the control panel with all parameter inputs"""
        # K value (number of clusters)
        ttk.Label(self.left_frame, text="Number of Clusters (k):").grid(column=0, row=0, padx=5, pady=5, sticky=tk.W)
        self.k_var = tk.IntVar(value=self.k)
        self.k_slider = ttk.Scale(self.left_frame, from_=2, to=9, orient=tk.HORIZONTAL, 
                                  variable=self.k_var, length=200, command=self.update_k_display)
        self.k_slider.grid(column=0, row=1, padx=5, pady=5, sticky=tk.W)
        self.k_display = ttk.Label(self.left_frame, text=str(self.k))
        self.k_display.grid(column=1, row=1, padx=5, pady=5, sticky=tk.W)
        
        # Circle diameter
        ttk.Label(self.left_frame, text="Circle Diameter:").grid(column=0, row=2, padx=5, pady=5, sticky=tk.W)
        self.diameter_var = tk.IntVar(value=self.circle_diameter)
        self.diameter_slider = ttk.Scale(self.left_frame, from_=10, to=300, orient=tk.HORIZONTAL, 
                                        variable=self.diameter_var, length=200, command=self.update_diameter_display)
        self.diameter_slider.grid(column=0, row=3, padx=5, pady=5, sticky=tk.W)
        self.diameter_display = ttk.Label(self.left_frame, text=str(self.circle_diameter))
        self.diameter_display.grid(column=1, row=3, padx=5, pady=5, sticky=tk.W)
        
        # Random diameter variation
        ttk.Label(self.left_frame, text="Random Diameter Variation:").grid(column=0, row=4, padx=5, pady=5, sticky=tk.W)
        self.diameter_random_var = tk.IntVar(value=self.diameter_random)
        self.diameter_random_slider = ttk.Scale(self.left_frame, from_=0, to=200, orient=tk.HORIZONTAL, 
                                               variable=self.diameter_random_var, length=200, command=self.update_diameter_random_display)
        self.diameter_random_slider.grid(column=0, row=5, padx=5, pady=5, sticky=tk.W)
        self.diameter_random_display = ttk.Label(self.left_frame, text=str(self.diameter_random))
        self.diameter_random_display.grid(column=1, row=5, padx=5, pady=5, sticky=tk.W)
        
        # Points per click
        ttk.Label(self.left_frame, text="Points per Click:").grid(column=0, row=6, padx=5, pady=5, sticky=tk.W)
        self.points_var = tk.IntVar(value=self.points_per_click)
        self.points_slider = ttk.Scale(self.left_frame, from_=1, to=50, orient=tk.HORIZONTAL, 
                                      variable=self.points_var, length=200, command=self.update_points_display)
        self.points_slider.grid(column=0, row=7, padx=5, pady=5, sticky=tk.W)
        self.points_display = ttk.Label(self.left_frame, text=str(self.points_per_click))
        self.points_display.grid(column=1, row=7, padx=5, pady=5, sticky=tk.W)
        
        # Show circles checkbox
        self.show_circles_var = tk.BooleanVar(value=self.show_circles)
        self.show_circles_check = ttk.Checkbutton(self.left_frame, text="Show Circles", 
                                                 variable=self.show_circles_var)
        self.show_circles_check.grid(column=0, row=8, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        # Action buttons
        self.button_frame = ttk.Frame(self.left_frame)
        self.button_frame.grid(column=0, row=9, columnspan=2, padx=5, pady=20, sticky=tk.W)
        
        self.clear_button = ttk.Button(self.button_frame, text="Clear Points", command=self.clear_points)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.kmeans_button = ttk.Button(self.button_frame, text="Run K-means", command=self.run_kmeans)
        self.kmeans_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(self.button_frame, text="Save Points", command=self.save_points)
        self.save_button.pack(side=tk.LEFT, padx=5)

    def create_drawing_canvas(self):
        """Create the canvas for drawing points"""
        # Canvas frame with more visible border
        self.canvas_frame = ttk.LabelFrame(self.center_frame, text="Drawing Canvas")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        # Add an extra frame inside for additional border effect
        self.canvas_border_frame = ttk.Frame(self.canvas_frame, borderwidth=3, relief=tk.GROOVE)
        self.canvas_border_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas for drawing - with visible border
        self.canvas = tk.Canvas(self.canvas_border_frame, width=self.canvas_width, height=self.canvas_height, bg="white", bd=2, relief=tk.RIDGE)
        self.canvas.pack(padx=5, pady=5)  # Don't expand to ensure exact dimensions
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Initialize the image for the canvas
        self.update_canvas_image()

    def update_canvas_image(self):
        """Convert the OpenCV image to a format compatible with Tkinter"""
        self.rgb_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(self.rgb_img))
        
        # Clear previous image and create new one that fills entire canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def on_canvas_click(self, event):
        """Handle canvas click events to generate points"""
        # Get current parameters
        self.k = int(self.k_var.get())
        self.circle_diameter = int(self.diameter_var.get())
        self.diameter_random = int(self.diameter_random_var.get())
        self.points_per_click = int(self.points_var.get())
        self.show_circles = self.show_circles_var.get()
        
        # Generate random points in a circle
        new_points = self.generate_random_points_in_circle(event.x, event.y, 
                                                        self.circle_diameter, 
                                                        self.diameter_random,
                                                        self.points_per_click)
        
        # Add the points to our list
        self.points.extend(new_points)
        
        # Draw the generated points
        for point_x, point_y in new_points:
            cv2.circle(self.img, (point_x, point_y), 3, (0, 0, 0), -1)
        
        # Update the canvas
        self.update_canvas_image()
        
        # Update status
        self.status_var.set(f"Added {self.points_per_click} points. Total points: {len(self.points)}")

    def generate_random_points_in_circle(self, center_x, center_y, base_diameter, random_variation, num_points):
        """Generate random points within a circle with random diameter variation"""
        new_points = []
        
        for _ in range(num_points):
            # Generate a random diameter for each point by adding random variation
            actual_diameter = base_diameter + random.uniform(0, random_variation)
            radius = actual_diameter / 2
            
            # Generate random angle and distance from center
            angle = random.uniform(0, 2 * np.pi)
            # Use square root for uniform distribution across the circle area
            distance = random.uniform(0, 1) ** 0.5 * radius
            
            # Calculate coordinates
            x = int(center_x + distance * np.cos(angle))
            y = int(center_y + distance * np.sin(angle))
            
            # Make sure the point is within the image boundaries
            x = max(0, min(x, self.img.shape[1] - 1))
            y = max(0, min(y, self.img.shape[0] - 1))
            
            new_points.append([x, y])
        
        # Draw outer boundary circle (using base_diameter + random_variation to show max possible range)
        if self.show_circles:
            max_radius = int((base_diameter + random_variation) / 2)
            cv2.circle(self.img, (center_x, center_y), max_radius, (220, 220, 220), 1)
            # Also draw the base diameter circle in a different color
            cv2.circle(self.img, (center_x, center_y), int(base_diameter / 2), (180, 180, 180), 1)
        
        return new_points

    def clear_points(self):
        """Clear all points and reset the canvas"""
        self.points = []
        self.img = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255  # Reset to white background
        self.update_canvas_image()
        self.status_var.set("All points cleared. Ready for new points.")

    def save_points(self):
        """Save the points to a numpy file in the points folder"""
        if len(self.points) == 0:
            self.status_var.set("No points to save.")
            return
            
        # Generate a default filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"points_{timestamp}.npy"
        
        # Ask user for filename
        file_path = filedialog.asksaveasfilename(
            initialdir=self.points_dir,
            initialfile=default_filename,
            defaultextension=".npy",
            filetypes=[("NumPy Files", "*.npy")]
        )
        
        if not file_path:
            # User cancelled
            return
            
        # Convert points to numpy array and save
        points_array = np.array(self.points)
        np.save(file_path, points_array)
        
        # Update status
        filename = os.path.basename(file_path)
        self.status_var.set(f"Saved {len(self.points)} points to {filename}")

    def run_kmeans(self):
        """Apply K-means clustering to the collected points"""
        # Update k from the slider
        self.k = int(self.k_var.get())
        
        if len(self.points) < self.k:
            self.status_var.set(f"Need at least {self.k} points to perform clustering with k={self.k}")
            return
        
        # Show status
        self.status_var.set("Running K-means clustering...")
        
        # Use a thread to prevent UI freezing
        threading.Thread(target=self._perform_kmeans_clustering).start()

    def _perform_kmeans_clustering(self):
        """Run K-means clustering in a separate thread to avoid freezing the UI"""
        # Convert points to the format required by K-means
        data = np.float32(self.points)
        
        # Apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, self.labels, self.centers = cv2.kmeans(data, self.k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert all to integer values
        self.centers = np.uint16(self.centers)
        
        # Create a visualization image
        clustered_img = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        
        # Colors for each cluster (BGR format for OpenCV)
        colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 0),    # Dark Blue
            (0, 128, 0),    # Dark Green
            (0, 0, 128)     # Dark Red
        ]
        
        # Draw each point with its cluster color
        for i, point in enumerate(self.points):
            x, y = point
            cluster_idx = self.labels[i][0]
            cv2.circle(clustered_img, (x, y), 3, colors[cluster_idx], -1)
        
        # Draw the centers of the clusters
        for i, center in enumerate(self.centers):
            x, y = center
            cv2.circle(clustered_img, (x, y), 10, colors[i], 2)
            cv2.putText(clustered_img, f"Cluster {i+1}", (x + 15, y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
        
        # Display in a new window and create a matplotlib plot
        self.visualize_kmeans_results(clustered_img, data)
        
        # Update status
        self.root.after(0, lambda: self.status_var.set("K-means clustering completed successfully."))

    def visualize_kmeans_results(self, clustered_img, data):
        """Create visualization of the clustering results"""
        # Create a new top level window for results
        results_window = tk.Toplevel(self.root)
        results_window.title("K-means Clustering Results")
        
        # Set the window to a reasonable size
        results_window.geometry("1000x800")
        
        # Create a button frame at the bottom
        button_frame = ttk.Frame(results_window)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Add a close button
        close_button = ttk.Button(button_frame, text="Close Results", command=results_window.destroy)
        close_button.pack(side=tk.RIGHT, padx=10)
        
        # Add save results button
        save_results_button = ttk.Button(button_frame, text="Save Clustering Results", 
                                        command=lambda: self.save_clustering_results())
        save_results_button.pack(side=tk.RIGHT, padx=10)
        
        # Add a key binding to close with Escape key
        results_window.bind('<Escape>', lambda event: results_window.destroy())
        
        # Container frames
        left_vis = ttk.Frame(results_window)
        left_vis.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        right_vis = ttk.Frame(results_window)
        right_vis.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Display the clustered image on the left
        cv_img = cv2.cvtColor(clustered_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        img_label = ttk.Label(left_vis)
        img_label.pack(fill=tk.BOTH, expand=True)
        img_label.image = tk_img  # Keep a reference to prevent garbage collection
        img_label.configure(image=tk_img)
        
        # Create a matplotlib figure for the right side
        fig = Figure(figsize=(8, 6), dpi=100)
        plot_ax = fig.add_subplot(111)
        
        # Colors for each cluster (RGB format for matplotlib)
        mpl_colors = [
            'blue', 'green', 'red', 'cyan', 'magenta', 'yellow',
            'darkblue', 'darkgreen', 'darkred'
        ]
        
        # Plot each cluster with a different color
        for i in range(self.k):
            cluster_points = data[self.labels.ravel() == i]
            plot_ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=80, 
                        c=mpl_colors[i], label=f'Cluster {i+1}')
        
        # Plot the centers
        plot_ax.scatter(self.centers[:, 0], self.centers[:, 1], s=300, c='black', marker='X', label='Centroids')
        
        plot_ax.set_title(f'K-means Clustering with k={self.k}')
        plot_ax.set_xlabel('X')
        plot_ax.set_ylabel('Y')
        plot_ax.legend()
        plot_ax.invert_yaxis()  # Invert Y-axis to match coordinate system
        
        # Embed the matplotlib figure in tkinter
        canvas = FigureCanvasTkAgg(fig, right_vis)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def save_clustering_results(self):
        """Save clustering results to a numpy file"""
        if self.labels is None or self.centers is None:
            self.status_var.set("No clustering results to save.")
            return
            
        # Create clustering results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clustering_results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Generate a default filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"kmeans_k{self.k}_{timestamp}"
        
        # Ask user for filename
        file_path = filedialog.asksaveasfilename(
            initialdir=results_dir,
            initialfile=default_filename,
            defaultextension=".npz",
            filetypes=[("NumPy Compressed Files", "*.npz")]
        )
        
        if not file_path:
            # User cancelled
            return
            
        # Save clustering results (points, labels, and centers)
        np.savez(file_path, 
                points=np.array(self.points), 
                labels=self.labels, 
                centers=self.centers,
                k=self.k)
        
        # Update status
        filename = os.path.basename(file_path)
        self.status_var.set(f"Saved clustering results to {filename}")
