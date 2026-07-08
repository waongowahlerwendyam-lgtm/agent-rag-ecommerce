import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json

# Configuration de la page
st.set_page_config(page_title="🛒 Assistant E-commerce", page_icon="🛍️")
st.title("🤖 Agent RAG - E-commerce Intelligent")

# Initialiser les sessions
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Charger les modèles (une seule fois)
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="langchain")
    return model, collection

model, collection = load_models()

# 2. Fonction de recherche
def rechercher(question, k=3):
    question_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )
    return results

# 3. Fonction Groq (avec la bonne clé)
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

# 4. Interface utilisateur
st.sidebar.title("Modes")
mode = st.sidebar.radio("Choisis ton mode :", ["Question", "Recommandation", "Prix", "Commande"])

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
    # Afficher la question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Préparer la réponse
    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche en cours..."):
            results = rechercher(prompt)
            contextes = results["documents"][0]
            contexte_complet = "\n".join(contextes)
            
            # Appel à l'IA
            reponse = interroger_groq(contexte_complet, prompt)
            
            st.markdown(reponse)
            
            # Afficher les sources
            with st.expander("📄 Sources utilisées"):
                for i, doc in enumerate(contextes):
                    st.caption(f"Source {i+1}: {doc[:200]}...")
            
            # Sauvegarder dans l'historique
            st.session_state.messages.append({
                "role": "assistant", 
                "content": reponse,
                "sources": contextes
            })