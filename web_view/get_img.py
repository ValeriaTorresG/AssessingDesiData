import os

directorio = "../data/plots/spectra/20211130"
images = [img for img in os.listdir(directorio)
            if img.lower().endswith(('.png'))]
text = ''

for image in images:
    text += f"<img src='{directorio}/{image}' alt='{image}''>\n"

with open("galery.txt", "w", encoding="utf-8") as file:
    file.write(text)

print("img saved in txt")