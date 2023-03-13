# opencv
import os
from helpers.ImageReader import ImageReader


imageReader = ImageReader()
dir_path = os.path.dirname(os.path.realpath(__file__))
INPUT_FILE = f"{dir_path}\\storage_imgs\\Lista_de_credores_Americanas.pdf"
# path = "expense_audit\\storage_imgs\\RC301906B.jpeg"

# pil_img = imageReader.base64_to_img(path)
# pil_img.save("expense_audit\\storage_imgs\\comprovante_2.jpeg")
text_list = imageReader.img_to_text(INPUT_FILE)
dicts = imageReader.dict_formater(text_list)
print(dicts)
