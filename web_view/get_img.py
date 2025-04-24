import os

directorio = "../plots/spectra/20211110"
images = [img for img in os.listdir(directorio)
            if img.lower().endswith(('.png'))]
text = ''

for image in images:
    text += f"<img src='{directorio}/{image}' alt='{image}''>\n"

with open("galeria.txt", "w", encoding="utf-8") as file:
    file.write(text)

print("img saved in txt")