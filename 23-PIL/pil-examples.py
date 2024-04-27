from PIL import Image

with Image.open('./img.png') as img:
    img.save('./new-img.png')

    smol_img = img.resize((300,300))
    thumbnailed_img = img.thumbnail((300,300))
    smol_img.save('smol-img.png')


