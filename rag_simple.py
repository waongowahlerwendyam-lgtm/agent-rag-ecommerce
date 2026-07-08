import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json

print("🔄 Chargement du modèle d'embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("📂 Connexion à ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="langchain")

def rechercher(question, k=3):
    question_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )
    return results

def interroger_groq(contexte, question):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }
    prompt = f"""Tu es un assistant e-commerce. Réponds UNIQUEMENT en utilisant le CONTEXTE ci-dessous.
Si la réponse n'est pas dans le CONTEXTE, réponds exactement : "Je ne sais pas, l'information n'est pas dans ma base."

CONTEXTE :
{contexte}

QUESTION : {question}
REPONSE :"""
    
    # UTILISATION DU NOUVEAU MODELE (le plus récent et stable)
    data = {
        "model": "llama-3.3-70b-versatile",  # <--- Changement ici
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    # SI ERREUR (clé invalide, limite, etc.) on affiche le détail
    if response.status_code != 200:
        print(f"❌ Erreur API Groq (Code {response.status_code}):")
        print(response.text)
        return "Désolé, l'IA n'a pas pu répondre à cause d'une erreur technique."
    
    try:
        return response.json()["choices"][0]["message"]["content"]
    except KeyError:
        print("❌ Erreur de parsing de la réponse Groq:")
        print(response.text)
        return "Désolé, une erreur interne est survenue."

if __name__ == "__main__":
    question = "Qui est le président du Burkina Faso ?"
    print(f"🧑 Question : {question}")
    
    results = rechercher(question)
    contextes = results["documents"][0]
    contexte_complet = "\n".join(contextes)
    
    reponse = interroger_groq(contexte_complet, question)
    print("\n🤖 Réponse de l'agent :")
    print(reponse)
    
    print("\n📄 Sources utilisees :")
    for i, doc in enumerate(contextes):
        print(f"Source {i+1}: {doc[:150]}...")