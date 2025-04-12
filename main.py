"""
Main entry point for the K-means Clustering Application (Version 2)
This version includes enhanced filtering capabilities for point connections.
"""

import tkinter as tk
import sys
import os

# Add the parent directory to the system path to allow relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core application
from version2.core.kmeans_app import KMeansClusteringApp

# Import extensions
from version2.extensions.connections_extension import ConnectionsExtension

def extend_app(app):
    """
    Extend the base application with additional functionality
    
    Args:
        app: The KMeansClusteringApp instance
    """
    # Keep track of loaded extensions
    extensions = {}
    
    # Add connections extension
    connections_ext = ConnectionsExtension(app)
    extensions['connections'] = connections_ext
    
    # Save references to extensions
    app.extensions = extensions
    
    # Update status
    app.status_var.set("Application loaded with connection filters. Ready to generate points.")

def main():
    """Run the application"""
    # Create the main window
    root = tk.Tk()
    
    # Set window icon and other properties if needed
    root.title("K-means Clustering Tool v2")
    
    # Create the application
    app = KMeansClusteringApp(root)
    
    # Extend with additional functionality
    extend_app(app)
    
    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()
