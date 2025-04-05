from pypdf import PdfReader
from pymongo import MongoClient
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ollama
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def connect_to_mongo():
    MONGO_URI = "mongodb+srv://ishikajindal062:DoFg3P167bgWybor@cluster0.yu9abni.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    DATABASE_NAME = "test_database"
    COLLECTION_NAME = "large_strings"
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    return client, collection

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    corpus_list = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            cleaned_text = text.encode("utf-8", "ignore").decode("utf-8")
            corpus_list.append(cleaned_text)
    return corpus_list

def store_text_in_mongo(collection, corpus_list):
    DOCUMENT_ID = "LargeTextList"
    CHUNK_SIZE = 900000
    for index, i in enumerate(range(0, len(corpus_list), CHUNK_SIZE)):
        chunk = corpus_list[i:i + CHUNK_SIZE]
        document = {
            "document_id": DOCUMENT_ID,
            "chunk_index": index,
            "content": chunk
        }
        collection.insert_one(document)

def retrieve_text_from_mongo(collection):
    DOCUMENT_ID = "LargeTextList"
    documents = collection.find({"document_id": DOCUMENT_ID}).sort("chunk_index")
    corpus_list = []
    for doc in documents:
        corpus_list.extend(doc["content"])
    return corpus_list

def rank_documents_bm25(corpus_list, query, k=2):
    if not corpus_list:
        return []
    indexed_corpus = {i: doc for i, doc in enumerate(corpus_list)}
    tokenized_corpus = [" ".join(map(str, bm25s.tokenize(doc))) for doc in corpus_list]
    retriever = bm25s.BM25(corpus=tokenized_corpus)
    retriever.index(tokenized_corpus)
    tokenized_query = bm25s.tokenize(query)
    results, scores = retriever.retrieve(tokenized_query, k=k)
    # ranked_docs = [(int(results[i][0][0]), scores[i][0]) for i in range(min(k, len(results))) if int(results[i][0][0]) in indexed_corpus]
    ranked_docs = []
    for i in range(min(k, len(results))):
        try:
            doc_index = int(results[i][0]) if isinstance(results[i][0], (int, np.integer)) else int(results[i][0][0])
            if doc_index in indexed_corpus:
                ranked_docs.append((doc_index, scores[i][0]))
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid entry {results[i][0]}: {e}")

    print("Results type:", type(results))
    print("Results shape (if NumPy array):", results.shape if hasattr(results, "shape") else "N/A")
    print("Results content:", results[:5])  # Print first few entries


    return ranked_docs

def get_full_text_of_top_document(retrieved_corpus, ranked_docs):
    if not ranked_docs:
        return ""
    top_doc_index = int(ranked_docs[0][0])
    return retrieved_corpus[top_doc_index] if top_doc_index < len(retrieved_corpus) else ""

def retrieve_most_relevant_text(corpus_list, query, top_k=2):
    """Retrieve the most relevant corpus entry based on a query using TF-IDF"""
    if not corpus_list:
        print("Corpus is empty.")
        return []
    
    # Convert text into TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    corpus_vectors = vectorizer.fit_transform(corpus_list)
    
    # Transform query into vector
    query_vector = vectorizer.transform([query])
    
    # Compute similarity scores
    similarity_scores = cosine_similarity(query_vector, corpus_vectors).flatten()
    
    # Get top-k indices
    top_indices = similarity_scores.argsort()[-top_k:][::-1]  
    return [(i, similarity_scores[i], corpus_list[i]) for i in top_indices]

def query_llama_with_context(context, question):
    prompt = f"""
    You are an AI assistant. Use the following context to answer the question accurately.
    
    ### Context:
    {context}
    
    ### Question:
    {question}
    
    Provide a concise and clear response.
    """
    
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    
    return response['message']['content']

def compute_bleu(reference, candidate):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    
    smoothie = SmoothingFunction().method1
    score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    
    return score

def main():
    pdf_path = r"C:\Users\HP\Downloads\Machine-Learning-with-Scikit-Learn.pdf"
    client, collection = connect_to_mongo()
    corpus_list = extract_text_from_pdf(pdf_path)
    store_text_in_mongo(collection, corpus_list)
    retrieved_corpus = retrieve_text_from_mongo(collection)
    # print(retrieved_corpus[10])
    query = "what is supervised machine learning"
    top_results = retrieve_most_relevant_text(retrieved_corpus, query, top_k=2)
    for i, (idx, score, text) in enumerate(top_results):
        print(f"\nTop {i+1} Match (Score: {score:.4f}):\n{text[:500]}")
    # print(top_results)
    response = query_llama_with_context(top_results[0][2], query)
    print("Llama 3 Response:\n", response)
    answer ="Supervised learning is a type of machine learning in which our algorithms are trained using well-labeled training data, and machines predict the output based on that data. Labeled data indicates that the input data has already been tagged with the appropriate output. Basically, it is the task of learning a function that maps the input set and returns an output. Some of its examples are: Linear Regression, Logistic Regression, KNN, etc."
    bleu_score = compute_bleu(answer, response)
    print(f"BLEU Score: {bleu_score:.4f}")

if __name__ == "__main__":
    top_text=main()
