# Projeto linkedin_crawler
---

# Para rodar o projeto na sua máquina:

1. Crie o ambiente virtual para o projeto

- `python3 -m venv .venv source .venv/bin/activate`

2. Instale as dependências

- `pip install -r .\requirements.txt`

3. Configure as variáveis de ambiente

- `.env.example`


# MongoDB

Para a realização deste projeto, utilizaremos um banco de dados chamado `companies_linkedin`, os links das empresas serão armazenados em uma coleção chamada `companies_links` e as empresas serão armazenadas em uma coleção chamada `companies`. Já existem algumas funções prontas no arquivo `crawler_linkedin/database.py` que te auxiliarão no desenvolvimento.

Lembre-se de que o mongoDB utilizará por padrão a porta 27017. Se já houver outro serviço utilizando esta porta, considere desativá-lo.

---
