import pymongo
from pymongo import MongoClient
MONGO_URI = "mongodb+srv://ishikajindal062:DoFg3P167bgWybor@cluster0.yu9abni.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

client = MongoClient(MONGO_URI)
db = client["test_database"] 
collection = db["large_strings"]
large_text = "A" * 10**6  
document = {"name": "LargeText1", "content": large_text}
insert_result = collection.insert_one(document)

print(f"Inserted document ID: {insert_result.inserted_id}")
retrieved_doc = collection.find_one({"name": "LargeText1"})
print(f"Retrieved document size: {len(retrieved_doc['content'])} characters")

client.close()