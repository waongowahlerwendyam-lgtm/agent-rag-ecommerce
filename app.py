import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
import pandas as pd
from streamlit_option_menu import option_menu

# ============================================
# 1. CONFIGURATION DE LA PAGE (Style pro)
# ============================================
st.set_page_config(
    page_title="Agent RAG Pro - E-commerce",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un look moderne
st.markdown("""
<style>
    /* Fond général */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Titre principal */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        padding: 1rem 0;
        text-align: center;
        border-bottom: 3px solid #4a90e2;
        margin-bottom: 2rem;
    }
    .main-header span {
        color: #4a90e2;
    }
    /* Messages du chat */
    .stChatMessage {
        border-radius: 18px;
        padding: 12px 20px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    /* Messages utilisateur (alignés à droite) */
    div[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]:first-child) {
        background-color: #4a90e2;
        color: white !important;
        border-bottom-right-radius: 4px;
    }
    /* Messages assistant (alignés à gauche) */
    div[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]:nth-child(2)) {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-bottom-left-radius: 4px;
    }
    /* Barre latérale */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #e8e8e8;
        padding-top: 2rem;
    }
    /* Style des sources */
    .source-box {
        background-color: #f1f3f5;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
        border-left: 4px solid #4a90e2;
    }
    /* Badge de mode */
    .mode-badge {
        display: inline-block;
        background-color: #4a90e2;
        color: white;
        padding: 4px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 2. FONCTIONS DE BASE (inchangées)
# ============================================

def creer_base_vectorielle():
    """Crée la base ChromaDB à partir de data.csv"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        try:
            collection = client.get_collection(name="avis_produits")
            return client, collection
        except chromadb.errors.NotFoundError:
            collection = client.create_collection(name="avis_produits")
            df = pd.read_csv("data.csv")
            documents, metadatas, ids = [], [], []
            for idx, row in df.iterrows():
                texte = f"Produit : {row['produit_nom']}. Prix : {row['prix']}. Avis : {row['avis_texte']} Note : {row['note']}/5"
                documents.append(texte)
                metadatas.append({"produit": row['produit_nom'], "prix": row['prix'], "note": str(row['note'])})
                ids.append(f"id_{idx}")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(documents).tolist()
            collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
            return client, collection
    except Exception as e:
        st.error(f"❌ Erreur : {e}")
        return None, None

@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client, collection = creer_base_vectorielle()
    return model, client, collection

model, client, collection = load_models()

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

def interroger_groq(contexte, question):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }
    prompt = f"""Tu es un assistant e-commerce professionnel. 
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

# ============================================
# 3. INTERFACE UTILISATEUR (Design Pro)
# ============================================

# --- En-tête ---
st.markdown('<div class="main-header">🛒 Assistant <span>RAG</span> · E-commerce</div>', unsafe_allow_html=True)

# --- Sidebar (Menu élégant) ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/shopping-bag.png", width=60)
    st.markdown("### 🧠 Tableau de bord")
    
    selected = option_menu(
        menu_title=None,
        options=["Question", "Recommandation", "Prix", "Commande"],
        icons=["chat-dots", "star", "tag", "truck"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#4a90e2", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#4a90e2", "color": "white"},
        }
    )
    
    st.markdown("---")
    st.caption("🔍 Mode actif : **" + selected + "**")
    st.caption("📊 Base : " + str(len(collection.get()['ids']) if collection else 0) + " documents")
    st.caption("⚡ Powered by Groq + Llama 3")

# --- Zone de chat ---
st.markdown(f'<span class="mode-badge">💬 Mode {selected}</span>', unsafe_allow_html=True)

# Initialiser l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages avec un design amélioré
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📄 Voir les sources utilisées"):
                for i, s in enumerate(msg["sources"]):
                    st.markdown(f'<div class="source-box">📌 Source {i+1} : {s[:200]}...</div>', unsafe_allow_html=True)

# Zone de saisie
if prompt := st.chat_input("💬 Posez votre question sur un produit..."):
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche en cours..."):
            if collection is None:
                st.error("❌ La base vectorielle n'a pas pu être créée.")
            else:
                results = rechercher(prompt)
                if results and results["documents"]:
                    contextes = results["documents"][0]
                    contexte_complet = "\n".join(contextes)
                    reponse = interroger_groq(contexte_complet, prompt)
                    st.markdown(reponse)
                    
                    with st.expander("📄 Voir les sources utilisées"):
                        for i, doc in enumerate(contextes):
                            st.markdown(f'<div class="source-box">📌 Source {i+1} : {doc[:200]}...</div>', unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": reponse,
                        "sources": contextes
                    })
                else:
                    st.write("😕 Aucune source trouvée. Essayez une autre question.")

# --- Pied de page ---
st.markdown('<div class="footer">🔒 Projet Data Science 2026 · Agent RAG Intelligent · IFOAD UJKZ</div>', unsafe_allow_html=True)
