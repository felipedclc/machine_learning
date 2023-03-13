arr = [
    "Iata-Azul-Latan-Btc-Acordo",
    "Agreement-Custumer",
    "iata-azul-latam-btc",
    "rextur_iata_latam",
    "iata",
    "rextur_iata",
]

formated = ""
for item in arr:
    formated += f"ft.consolidator = {item.upper()} or "

    # formated.append(f"ft.consolidator = {item.upper()}")

print(formated)
