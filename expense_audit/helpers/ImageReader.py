import io
import numpy
import cv2
import base64
import pytesseract
from PIL import Image
from helpers.aux_func import (
    put_date_on_json_if_exists,
    put_cnpj_on_json_if_exists,
    put_amount_on_json_if_exists,
    # put_items_on_json_if_exists,
    put_type_on_json_if_exists,
    improve_search_accuracy,
    get_un_values,
)

# from pytesseract import Output
import matplotlib.pyplot as plt

config = "--oem 3 --psm 6"
lang = "eng+por"
# output = Output.DICT


class ImageReader:
    @classmethod
    def base64_to_img(cls, path):
        byte_data = open(path, "rb").read()
        byte_stream = base64.b64decode(byte_data)
        pil_img = Image.open(io.BytesIO(byte_stream))  # .convert("RGB")
        open_cv_image = numpy.array(pil_img)
        return open_cv_image[:, :, ::-1].copy()
        # return pil_img

    @classmethod
    def img_to_text(cls, pil_img):
        conf_text = []
        image1 = cv2.imread(pil_img)
        image = improve_search_accuracy(image1)
        extracted_text = pytesseract.image_to_string(
            image,
            lang=lang,
            config=config,  # output_type=output
        )

        # plt.imshow(image)
        # plt.show()
        splits = extracted_text.splitlines()
        for row in splits:
            if not row.isspace() and len(row) > 0:
                conf_text.append(row)
        return conf_text

    @classmethod
    def dict_formater(cls, text_list):
        json = {
            "text": text_list,
            "vendor": {
                "name": None,
                "document": None,
            },
            "data": {"type": None, "date": None, "total_amount": None},
            "items": [{"description": None, "total": None}],
        }
        # print(text_list)
        for text in text_list:
            put_date_on_json_if_exists(text, json)
            put_cnpj_on_json_if_exists(text, json)
            put_type_on_json_if_exists(text, json)
            put_amount_on_json_if_exists(text, json)

        json["items"] = get_un_values(text_list)
        json["vendor"]["name"] = text_list[0] + "" + text_list[1]
        return json
