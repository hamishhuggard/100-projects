#!/usr/bin/env python3
"""
Image Resizer Script

This script resizes an image to 1000px width while maintaining aspect ratio.
The original file is overwritten with the resized version.

Usage: python resize_image.py <image_path>
"""

import sys
import os
from PIL import Image

def resize_image(image_path, target_width=1000):
    """
    Resize an image to the specified width while maintaining aspect ratio.
    
    Args:
        image_path (str): Path to the image file
        target_width (int): Target width in pixels (default: 800)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Calculate new height maintaining aspect ratio
            width, height = img.size
            aspect_ratio = height / width
            new_height = int(target_width * aspect_ratio)
            
            # Resize the image
            resized_img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
            
            # Save the resized image, overwriting the original
            resized_img.save(image_path, quality=95, optimize=True)
            
            print(f"✓ Successfully resized {image_path}")
            print(f"  Original size: {width}x{height}")
            print(f"  New size: {target_width}x{new_height}")
            
            return True
            
    except FileNotFoundError:
        print(f"✗ Error: File '{image_path}' not found")
        return False
    except Exception as e:
        print(f"✗ Error processing image: {e}")
        return False

def main():
    """Main function to handle command line arguments and execute the resize operation."""
    
    # Check if image path is provided
    if len(sys.argv) != 2:
        print("Usage: python resize_image.py <image_path>")
        print("Example: python resize_image.py photo.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"✗ Error: File '{image_path}' does not exist")
        sys.exit(1)
    
    # Check if it's a file (not a directory)
    if not os.path.isfile(image_path):
        print(f"✗ Error: '{image_path}' is not a file")
        sys.exit(1)
    
    # Get file extension to check if it's an image
    file_ext = os.path.splitext(image_path)[1].lower()
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    if file_ext not in supported_formats:
        print(f"✗ Warning: File extension '{file_ext}' may not be supported")
        print(f"  Supported formats: {', '.join(supported_formats)}")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Perform the resize operation
    success = resize_image(image_path)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 