import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
import pandas as pd

# Configuration de la page
st.set_page_config(page_title="🛒 Assistant E-commerce", page_icon="🛍️")
st.title("🤖 Agent RAG - E-commerce Intelligent")

# 1. Fonction pour créer la base vectorielle (si elle n'existe pas)
def creer_base_vectorielle():
    """Crée la base ChromaDB à partir de data.csv si elle n'existe pas"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        # Vérifier si la collection existe déjà
        try:
            collection = client.get_collection(name="avis_produits")
            return client, collection
        except chromadb.errors.NotFoundError:
            # Créer la collection si elle n'existe pas
            collection = client.create_collection(name="avis_produits")
            
            # Charger le CSV
            df = pd.read_csv("data.csv")
            documents = []
            metadatas = []
            ids = []
            
            for idx, row in df.iterrows():
                texte = f"Produit : {row['produit_nom']}. Prix : {row['prix']}. Avis : {row['avis_texte']} Note : {row['note']}/5"
                documents.append(texte)
                metadatas.append({
                    "produit": row['produit_nom'],
                    "prix": row['prix'],
                    "note": str(row['note'])
                })
                ids.append(f"id_{idx}")
            
            # Générer les embeddings
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(documents).tolist()
            
            # Ajouter à ChromaDB
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return client, collection
    except Exception as e:
        st.error(f"Erreur lors de la création de la base : {e}")
        return None, None

# 2. Charger les modèles (une seule fois)
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client, collection = creer_base_vectorielle()
    return model, client, collection

# 3. Initialisation
model, client, collection = load_models()

# 4. Fonction de recherche
def rechercher(question, k=3):
    if collection is None:
        return None
    question_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )
    return results

# 5. Fonction Groq
def interroger_groq(contexte, question):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }
    prompt = f"""Tu es un assistant e-commerce. 
Réponds UNIQUEMENT en utilisant le CONTEXTE fourni.
Si la réponse n'est pas dans le CONTEXTE, réponds exactement : "Je ne sais pas, l'information n'est pas dans ma base."

CONTEXTE : {contexte}
QUESTION : {question}
REPONSE :"""
    
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Erreur API: {response.status_code}"

# 6. Interface utilisateur
st.sidebar.title("Modes")
mode = st.sidebar.radio("Choisis ton mode :", ["Question", "Recommandation", "Prix", "Commande"])

# Initialiser l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📄 Voir les sources"):
                for s in msg["sources"]:
                    st.caption(f"- {s[:150]}...")

# Zone de saisie
if prompt := st.chat_input("Pose ta question sur un produit..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche en cours..."):
            if collection is None:
                st.error("La base vectorielle n'a pas pu être créée.")
            else:
                results = rechercher(prompt)
                if results and results["documents"]:
                    contextes = results["documents"][0]
                    contexte_complet = "\n".join(contextes)
                    reponse = interroger_groq(contexte_complet, prompt)
                    st.markdown(reponse)
                    
                    with st.expander("📄 Sources utilisées"):
                        for i, doc in enumerate(contextes):
                            st.caption(f"Source {i+1}: {doc[:200]}...")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": reponse,
                        "sources": contextes
                    })
                else:
                    st.write("Aucune source trouvée. Essaye une autre question.")
