import streamlit as st
import PyPDF2
import networkx as nx
import pandas as pd
from collections import Counter, defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go

class PDFAnalyzer:
    def __init__(self):
        self.documents = {}
        self.available_categories = ['Salud Pública', 'Decolonial', 'Metodología', 'Otros']
        
    def extract_text(self, pdf_file):
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error al leer PDF: {e}")
        return text
    
    def extract_key_concepts(self, text, top_n=20):
        # Regex para palabras en español
        words = re.findall(r'\b[a-zA-ZáéíóúÁÉÍÓÚñÑ]{3,}\b', text.lower())
        common_words = Counter(words).most_common(30)
        
        # TF-IDF
        sentences = re.split(r'[.!?]+', text)
        # Filtramos oraciones muy cortas
        sentences = [s for s in sentences if len(s) > 10]
        
        if not sentences:
            return {'entities': [], 'tfidf': []}

        vectorizer = TfidfVectorizer(max_features=top_n, stop_words=None, ngram_range=(1,2))
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.mean(axis=0).A1
        
        return {
            'entities': common_words[:15],
            'tfidf': sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        }

    def build_concept_map(self, concepts_list):
        G = nx.Graph()
        # Tomamos los top 15 para no saturar el grafo
        for concept, score in concepts_list[:15]:
            G.add_node(concept, size=score * 100) # Escalamos el tamaño
        
        nodes = list(G.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, min(i + 3, len(nodes))):
                G.add_edge(nodes[i], nodes[j])
        return G

def main():
    st.set_page_config(page_title="PDF Analyzer Académico", layout="wide")
    st.title("🔬 Analizador Académico de PDFs")

    # --- INICIALIZACIÓN DEL ESTADO ---
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = PDFAnalyzer()

    analyzer = st.session_state.analyzer

    # --- SIDEBAR ---
    st.sidebar.header("📁 Gestión de Documentos")
    uploaded_files = st.sidebar.file_uploader("Subir PDFs", type='pdf', accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in analyzer.documents:
                with st.spinner(f"Procesando {file.name}..."):
                    text = analyzer.extract_text(file)
                    concepts = analyzer.extract_key_concepts(text)
                    analyzer.documents[file.name] = {
                        'text': text,
                        'concepts': concepts,
                        'categories': [],
                        'notes': ''
                    }

    if analyzer.documents:
        doc_names = list(analyzer.documents.keys())
        selected_doc = st.sidebar.selectbox("Seleccionar documento", doc_names)
        
        # Categorías dinámicas
        with st.sidebar.expander("📝 Categorizar"):
            new_cat = st.text_input("Nueva categoría")
            if st.button("Añadir") and new_cat:
                if new_cat not in analyzer.available_categories:
                    analyzer.available_categories.append(new_cat)
            
            selected_cats = st.multiselect(
                "Categorías del doc", 
                analyzer.available_categories,
                default=analyzer.documents[selected_doc]['categories']
            )
            analyzer.documents[selected_doc]['categories'] = selected_cats
            
            notes = st.text_area("Notas", value=analyzer.documents[selected_doc]['notes'])
            analyzer.documents[selected_doc]['notes'] = notes

        # --- TABS PRINCIPALES ---
        tab1, tab2, tab3 = st.tabs(["📊 Análisis", "🗺️ Mapa Conceptual", "🔍 Cruzado"])

        with tab1:
            st.subheader(f"Análisis: {selected_doc}")
            concepts = analyzer.documents[selected_doc]['concepts']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top Conceptos (TF-IDF)**")
                df_tfidf = pd.DataFrame(concepts['tfidf'], columns=['Concepto', 'Relevancia'])
                st.dataframe(df_tfidf, use_container_width=True)
            
            with col2:
                st.write("**Frecuencia de Palabras**")
                df_ent = pd.DataFrame(concepts['entities'], columns=['Palabra', 'Cant'])
                st.bar_chart(df_ent.set_index('Palabra'))

        with tab2:
            st.subheader("Relación de Conceptos")
            # Unimos TF-IDF para el grafo
            G = analyzer.build_concept_map(concepts['tfidf'])
            
            pos = nx.spring_layout(G)
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), mode='lines')

            node_x, node_y, node_text = [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)

            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text',
                text=node_text, textposition="top center",
                marker=dict(size=20, color='SkyBlue', line_width=2)
            )

            fig = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Análisis Cruzado")
            if len(analyzer.documents) > 1:
                other_doc = st.selectbox("Comparar con", [d for d in doc_names if d != selected_doc])
                # Lógica simple de intersección
                set1 = set(c[0] for c in analyzer.documents[selected_doc]['concepts']['tfidf'][:10])
                set2 = set(c[0] for c in analyzer.documents[other_doc]['concepts']['tfidf'][:10])
                shared = set1.intersection(set2)
                
                if shared:
                    st.success(f"Conceptos compartidos: {', '.join(shared)}")
                else:
                    st.info("No se encontraron conceptos compartidos en el top 10.")
    else:
        st.info("👋 Por favor, sube uno o más archivos PDF en la barra lateral para comenzar.")

if __name__ == "__main__":
    main()
