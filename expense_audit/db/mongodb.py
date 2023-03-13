from pymongo import MongoClient
import copy

client = MongoClient(host="localhost", port=27017)
db = client.expense_audit  # nome do bd


def create_companies(data):
    db.companies.insert_many(copy.deepcopy(data))


def insert_or_update_links(company):
    db.companies_links.update_one(
        {"company_link": company["company_link"]},
        {"$set": company},
        upsert=True,
    ).upserted_id


def insert_or_update_companies(company):
    print("mongolink", company["link"])
    db.companies.update_one(
        {"link": company["link"]},
        {"$set": company},
        upsert=True,
    ).upserted_id


def find_companies():
    return list(db.companies.find({}, {"_id": False}))


def find_companies_links():
    return list(db.companies_links.find({}, {"_id": False}))


def search_companies(query):
    return list(db.companies.find(query))


def get_collection():
    return db.companies
