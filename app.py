import streamlit as st
import pypdf
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from collections import Counter
import re

# Configuración inicial
st.set_page_config(page_title="PDF Analyzer", layout="wide")

# --- LÓGICA DE PROCESAMIENTO ---
def get_clean_text(file):
    """Extrae texto de un PDF cargado."""
    reader = pypdf.PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + " "
    return text

def get_keywords(text):
    """Extrae palabras clave ignorando conectores comunes en español."""
    # Lista básica de stopwords (para no depender de sklearn)
    stopwords = {'que', 'este', 'esta', 'los', 'las', 'del', 'con', 'una', 'un', 'para', 'por', 'sobre'}
    # Limpieza básica
    words = re.findall(r'\b[a-záéíóúñ]{4,}\b', text.lower())
    filtered = [w for w in words if w not in stopwords]
    return Counter(filtered).most_common(20)

# --- ESTADO DE LA APP ---
if 'docs' not in st.session_state:
    st.session_state.docs = {}

# --- UI - SIDEBAR ---
st.sidebar.title("📚 Panel de Control")
uploaded_files = st.sidebar.file_uploader("Sube tus PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.docs:
            with st.spinner(f"Analizando {file.name}..."):
                raw_text = get_clean_text(file)
                keywords = get_keywords(raw_text)
                st.session_state.docs[file.name] = {
                    "text": raw_text,
                    "keywords": keywords,
                    "notes": ""
                }

# --- UI - CUERPO PRINCIPAL ---
st.title("🔬 Analizador Académico de PDFs")

if not st.session_state.docs:
    st.info("Sube archivos PDF en la barra lateral para comenzar.")
else:
    doc_list = list(st.session_state.docs.keys())
    selected_name = st.selectbox("Selecciona un documento para analizar:", doc_list)
    
    current_doc = st.session_state.docs[selected_name]

    tab1, tab2 = st.tabs(["📊 Conceptos", "🗺️ Mapa Conceptual"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Top Palabras")
            df = pd.DataFrame(current_doc['keywords'], columns=['Palabra', 'Frecuencia'])
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Notas de Investigación")
            st.session_state.docs[selected_name]['notes'] = st.text_area(
                "Escribe hallazgos aquí:", value=current_doc['notes'], height=200
            )

    with tab2:
        st.subheader("Red de Términos Relacionados")
        # Generar Grafo simple
        G = nx.Graph()
        terms = [k[0] for k in current_doc['keywords'][:12]]
        for i, term in enumerate(terms):
            G.add_node(term)
            if i > 0: G.add_edge(terms[i-1], term) # Conexión lineal simple para visualización
            
        # Posiciones para el gráfico
        pos = nx.spring_layout(G)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Crear figura Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray', width=1)))
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in G.nodes()],
            y=[pos[n][1] for n in G.nodes()],
            mode='markers+text',
            text=list(G.nodes()),
            textposition="top center",
            marker=dict(size=15, color='orange')
        ))
        fig.update_layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.divider()
if st.sidebar.button("Limpiar todo"):
    st.session_state.docs = {}
    st.rerun()
