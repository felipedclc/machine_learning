import re
import cv2
import numpy as np
from itertools import cycle

# import matplotlib.pyplot as plt


def get_un_values(text_list: list) -> list:
    lines_with_un = [{"description": None, "total": None}]
    index = 0
    for line in text_list:
        if "UN" in line:
            if bool(re.search(r"\d", line)):
                formated = re.sub(r"[@]", "0", line)
                get_digits = re.findall(r"\d+\,\d+", formated)
                to_float = [
                    round(float(n.replace(",", ".")), 2) for n in get_digits
                ]
                if len(to_float[-1:]) > 0:
                    lines_with_un.append(
                        {
                            "description": text_list[index - 1],
                            "total": to_float[-1:][0],
                        }
                    )

        index += 1
    # print(lines_with_un)
    return lines_with_un


def put_date_on_json_if_exists(text: str, json: dict) -> None:
    date = re.search(r"\d+[-/]\d+[-/]\d+", text)
    if date is not None:
        date = date.group(0)
        if len(date[6:]) <= 2:
            json["data"]["date"] = f"{date[:6]}20{date[6:]}"
        else:
            json["data"]["date"] = date


def fix_cnpj(cnpj: str) -> bool:
    LENGTH_CNPJ = 14
    cnpj = "".join([n for n in cnpj if n.isdigit()])
    if len(cnpj) != LENGTH_CNPJ:
        print("CNPJ não pode possuir mais de 14 digitos")
        return None

    cnpj_r = cnpj[::-1]
    for i in range(2, 0, -1):
        cnpj_enum = zip(cycle(range(2, 10)), cnpj_r[i:])
        dv = sum(map(lambda x: int(x[1]) * x[0], cnpj_enum)) * 10 % 11
        cnpj_r = f"{cnpj_r[:(i - 1)]}{str(dv)}{cnpj_r[i:]}"

    new_cnpj = cnpj_r[::-1]
    return f"{new_cnpj[:2]}.{new_cnpj[2:5]}.{new_cnpj[5:8]}/{new_cnpj[8:12]}-{new_cnpj[12:]}"


def put_cnpj_on_json_if_exists(text: str, json: dict) -> None:
    cnpj = re.search(r"\d{2}\.\d{3}\.\d{3}\/\d{4}-\d{2}", text)
    if cnpj is not None:
        cnpj = cnpj.group(0)
        json["vendor"]["cnpj"] = fix_cnpj(cnpj)


def put_amount_on_json_if_exists(text: str, json: dict) -> None:
    if "Valor a Pagar" in text:
        amount = re.search(r"\d+\,\d+", text)
        json["data"]["total_amount"] = amount.group(0)


def put_type_on_json_if_exists(text: str, json: dict) -> None:
    if text.find("nfce") != -1:
        json["data"]["type"] = "NFCE"


def improve_search_accuracy(img):
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    return img
