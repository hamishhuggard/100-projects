from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import glob
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
ORDER_FILE = 'image_order.txt'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

@app.route('/api/images', methods=['GET'])
def get_images():
    """Get list of all images in the images directory"""
    try:
        image_files = []
        for ext in ALLOWED_EXTENSIONS:
            pattern = os.path.join(UPLOAD_FOLDER, f'*.{ext}')
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(UPLOAD_FOLDER, f'*.{ext.upper()}')
            image_files.extend(glob.glob(pattern))
        
        # Extract just the filenames
        images = []
        for filepath in image_files:
            filename = os.path.basename(filepath)
            images.append({
                'filename': filename,
                'src': f'images/{filename}'
            })
        
        return jsonify({
            'success': True,
            'images': images
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to read images directory: {str(e)}'
        }), 500

@app.route('/api/order', methods=['GET'])
def get_order():
    """Get current image order from text file"""
    try:
        if os.path.exists(ORDER_FILE):
            with open(ORDER_FILE, 'r', encoding='utf-8') as f:
                order = [line.strip() for line in f.readlines() if line.strip()]
        else:
            order = []
        
        return jsonify({
            'success': True,
            'order': order
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to read order file: {str(e)}'
        }), 500

@app.route('/api/order', methods=['POST'])
def save_order():
    """Save image order to text file"""
    try:
        data = request.get_json()
        order = data.get('order', [])
        
        if not isinstance(order, list):
            return jsonify({
                'success': False,
                'error': 'Order must be an array'
            }), 400
        
        # Write order to file
        with open(ORDER_FILE, 'w', encoding='utf-8') as f:
            for filename in order:
                f.write(f'{filename}\n')
        
        return jsonify({
            'success': True,
            'message': 'Order saved successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to save order: {str(e)}'
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload new image files"""
    try:
        if 'images' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files uploaded'
            }), 400
        
        files = request.files.getlist('images')
        uploaded_files = []
        
        for file in files:
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                uploaded_files.append({
                    'filename': filename,
                    'src': f'images/{filename}'
                })
        
        if not uploaded_files:
            return jsonify({
                'success': False,
                'error': 'No valid image files uploaded'
            }), 400
        
        return jsonify({
            'success': True,
            'message': f'{len(uploaded_files)} image(s) uploaded successfully',
            'files': uploaded_files
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to upload files: {str(e)}'
        }), 500

@app.route('/api/images/<filename>', methods=['DELETE'])
def delete_image(filename):
    """Delete an image file"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        
        if not os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': 'Image deleted successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to delete file: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🎬 Video Editor Backend starting...")
    print(f"📁 Images directory: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"📄 Order file: {os.path.abspath(ORDER_FILE)}")
    app.run(debug=True, host='0.0.0.0', port=3000)
