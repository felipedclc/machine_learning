from itertools import cycle


""" def fix_cnpj(cnpj: str) -> bool:
    LENGTH_CNPJ = 14
    cnpj = ''.join([n for n in cnpj if n.isdigit()])
    if len(cnpj) != LENGTH_CNPJ:
        print('CNPJ não pode possuir mais de 14 digitos')
        return None

    cnpj_r = cnpj[::-1]
    for i in range(2, 0, -1):
        cnpj_enum = zip(cycle(range(2, 10)), cnpj_r[i:])
        dv = sum(map(lambda x: int(x[1]) * x[0], cnpj_enum)) * 10 % 11
        cnpj_r = f"{cnpj_r[:(i - 1)]}{str(dv)}{cnpj_r[i:]}"

    new_cnpj = cnpj_r[::-1]
    return f"{new_cnpj[:2]}.{new_cnpj[2:5]}.{new_cnpj[5:8]}/{new_cnpj[8:12]}-{new_cnpj[12:]}"


print('is_cnpj_valido', fix_cnpj('04.149.637/0002-66')) """


# list_nu = ['1,0000', '79,90', '79,90']
# for n in list_nu:
#     form = round(float(n.replace(',', '.')), 2)
#     print(form)


st = "FORMA PAGAMENTG = # £VALOR PAGO R$ a-nfce"
print(st.find("nfce"))
