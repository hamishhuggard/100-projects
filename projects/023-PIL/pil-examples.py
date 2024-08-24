from PIL import Image
from PIL import ImageFilter

with Image.open('./img.png') as img:
    img.save('./new-img.png')

    smol_img = img.resize((300,300))
    thumbnailed_img = img.thumbnail((300,300))
    cropped = img.crop((100, 100, 200, 300))
    img.rotate(90)
    img.transpose(Image.FLIP_LEFT_RIGHT)
    img.filter(ImageFilter.BLUR)

    smol_img.save('smol-img.png')


