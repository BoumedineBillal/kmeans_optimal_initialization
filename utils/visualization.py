import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def draw_connections_on_image(img, points, connections, min_x, min_y, max_x, max_y, 
                             width=None, height=None, padding=20, color=(255, 200, 100), thickness=1):
    """
    Draw connections between points on a copy of the given image
    
    Args:
        img: The image on which to draw
        points: Array of point coordinates
        connections: List of tuples with indices of points to connect
        min_x, min_y, max_x, max_y: Bounds of the data
        width, height: Optional dimensions to override image dimensions
        padding: Padding around the edges
        color: Line color as BGR tuple
        thickness: Line thickness
        
    Returns:
        A new image with connections drawn
    """
    # Create a copy of the image to draw on
    img_result = img.copy()
    
    if width is None or height is None:
        height, width = img.shape[:2]
    
    display_width = width - 2 * padding
    display_height = height - 2 * padding
    
    data_width = max_x - min_x
    data_height = max_y - min_y
    
    if data_width <= display_width and data_height <= display_height:
        # No scaling needed
        offset_x = padding - min_x
        offset_y = padding - min_y
        scale = 1.0
    else:
        # Calculate scale and offset for proper display
        scale_x = display_width / data_width if data_width > 0 else 1
        scale_y = display_height / data_height if data_height > 0 else 1
        scale = min(scale_x, scale_y)
        
        scaled_width = data_width * scale
        scaled_height = data_height * scale
        offset_x = padding + (display_width - scaled_width) / 2 - min_x * scale
        offset_y = padding + (display_height - scaled_height) / 2 - min_y * scale
    
    # Draw the connections
    for i, j in connections:
        # Calculate the display coordinates for both points
        x1 = int(points[i][0] * scale + offset_x)
        y1 = int(points[i][1] * scale + offset_y)
        x2 = int(points[j][0] * scale + offset_x)
        y2 = int(points[j][1] * scale + offset_y)
        
        # Draw a line connecting them
        cv2.line(img_result, (x1, y1), (x2, y2), color, thickness)
    
    return img_result

def create_connections_window(parent, points, connections, threshold, img=None, window_title="Point Connections", extra_info=None):
    """
    Create a new window displaying the points with connections
    
    Args:
        parent: Parent window
        points: Array of point coordinates
        connections: List of tuples with indices of points to connect
        threshold: The threshold distance used for connections
        img: Optional image to display (if None, will create a new image)
        window_title: Title for the window (default: "Point Connections")
        extra_info: Optional additional information to display in status
        
    Returns:
        The created window
    """
    if points is None or len(points) == 0:
        return None
    
    # Create a top level window
    connections_window = tk.Toplevel(parent)
    connections_window.title(f"{window_title} (Threshold: {threshold:.2f})")
    
    # Set the window to a reasonable size
    connections_window.geometry("800x600")
    
    # Calculate bounds of the data
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])
    
    # Create frame for the image
    frame = ttk.Frame(connections_window)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Use a copy of the provided image or create a blank one
    if img is not None:
        img_with_connections = img.copy()
        height, width = img.shape[:2]
    else:
        width, height = 800, 600
        img_with_connections = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw connections directly on the canvas the same way the original points were drawn
    padding = 20
    color = (255, 200, 100)  # Light blue color
    
    for i, j in connections:
        # Get the original point coordinates
        pt1 = points[i]
        pt2 = points[j]
        
        # Use the same coordinate system as the original points
        # This is the key part to ensure alignment
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        
        # Draw the line directly connecting the points
        cv2.line(img_with_connections, (x1, y1), (x2, y2), color, 1)
    
    # Convert the image for tkinter display
    rgb_img = cv2.cvtColor(img_with_connections, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    tk_img = ImageTk.PhotoImage(image=pil_img)
    
    # Create a label to display the image
    img_label = ttk.Label(frame)
    img_label.pack(fill=tk.BOTH, expand=True)
    img_label.image = tk_img  # Keep a reference to prevent garbage collection
    img_label.configure(image=tk_img)
    
    # Status text
    status_text = f"Found {len(connections)} connections between {len(points)} points (threshold: {threshold:.2f})"
    if extra_info:
        status_text += f" | {extra_info}"
    
    # Add status label
    status_label = ttk.Label(
        connections_window, 
        text=status_text
    )
    status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
    
    # Add a close button
    close_button = ttk.Button(connections_window, text="Close", command=connections_window.destroy)
    close_button.pack(side=tk.BOTTOM, pady=10)
    
    # Bind escape key to close the window
    connections_window.bind("<Escape>", lambda event: connections_window.destroy())
    
    return connections_window
