from flask import Flask, request, send_file
from PIL import Image
import io

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <title> Upload to rotate </title>
    <h1> Upload to rotate </h1>
    <form method=post enctype=multipart/form-data action="/rotate">
        <input type=file name=image>
        <input type=submit value=Updload>
    </form>
    '''

@app.route('/rotate', methods=['POST'])
def rotate():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400
    image = Image.open(file.stream)
    rotated = image.rotate(90)
    if rotated.mode == 'RGBA':
        rotated = rotated.convert('RGB')
    img_io = io.BytesIO()
    rotated.save(img_io, 'JPEG', quality=70)
    print('rotated image')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__=='__main__':
    app.run(debug=True, port=5005)

