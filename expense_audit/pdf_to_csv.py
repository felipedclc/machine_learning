import os
from nanonets import NANONETSOCR

# import json


model = NANONETSOCR()

# Authenticate
# This software is perpetually free :)
model.set_token("41c4be20-9cd9-11ed-a6be-6696fc1e8944")

dir_path = os.path.dirname(os.path.realpath(__file__))
INPUT_FILE = f"{dir_path}\\storage_imgs\\Lista_de_credores_Americanas.pdf"

# PDF / Image to Raw OCR Engine Output

# pred_json = model.convert_to_prediction(INPUT_FILE)
# print(json.dumps(pred_json, indent=2))

# PDF / Image to String

# string = model.convert_to_string(INPUT_FILE)
# print(string)

# PDF / Image to TXT File

# model.convert_to_txt(INPUT_FILE, output_file_name="OUTPUTNAME.txt")

# PDF / Image to Boxes
# each element contains predicted word and bounding box information
# bounding box information denotes the spatial position of each word in the file

# boxes = model.convert_to_boxes("test.png")
# for box in boxes:
#     print(box)

# PDF / Image to CSV
# This method extracts tables from your file and prints them in a .csv file.
# NOTE : This particular function is a trial offering 1000 pages of use.
# To use this at scale, please create your own model at app.nanonets.com --> New Model --> Tables.
model.convert_to_csv(
    INPUT_FILE, output_file_name="lista_credores_americanas.csv"
)
