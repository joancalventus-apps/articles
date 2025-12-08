import streamlit as st
import PyPDF2
import spacy
import networkx as nx
import pandas as pd
from collections import Counter, defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Cargar modelo spaCy español
@st.cache_resource
def load_nlp():
    return spacy.load("es_core_news_sm")

nlp = load_nlp()

class PDFAnalyzer:
    def __init__(self):
        self.documents = {}
        self.categories = {}
        self.tags = defaultdict(list)
        
    def extract_text(self, pdf_file):
        """Extrae texto de PDF"""
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def extract_key_concepts(self, text, top_n=20):
        """Extrae conceptos clave usando TF-IDF + entidades NER"""
        doc = nlp(text)
        
        # Entidades nombradas
        entities = [ent.text.lower() for ent in doc.ents 
                   if ent.label_ in ['PER', 'ORG', 'LOC', 'MISC']]
        
        # Sustantivos compuestos (términos técnicos)
        nouns = [chunk.text.lower() for chunk in doc.noun_chunks 
                if len(chunk.text.split()) <= 4]
        
        # TF-IDF sobre oraciones
        sentences = [sent.text for sent in doc.sents]
        vectorizer = TfidfVectorizer(max_features=top_n, stop_words='spanish', 
                                   ngram_range=(1,3))
        tfidf_matrix = vectorizer.fit_transform(sentences)
        feature_names = vectorizer.get_feature_names_out()
        
        return {
            'entities': Counter(entities).most_common(15),
            'nouns': Counter(nouns).most_common(15),
            'tfidf': [(name, score) for name, score in 
                     zip(feature_names, tfidf_matrix.mean(axis=0).A1)]
        }
    
    def build_concept_map(self, concepts, top_relations=10):
        """Construye grafo de conceptos"""
        G = nx.Graph()
        for concept, count in concepts[:top_relations]:
            G.add_node(concept, size=count, type='concept')
        # Conexiones basadas en co-ocurrencia aproximada
        for i in range(len(concepts)-1):
            for j in range(i+1, min(i+4, len(concepts))):
                G.add_edge(concepts[i][0], concepts[j][0], weight=0.5)
        return G

def main():
    st.set_page_config(page_title="PDF Analyzer Académico", layout="wide")
    
    st.title("🔬 Analizador Académico de PDFs")
    st.markdown("**Extracción de conceptos, mapas conceptuales, categorización y análisis cruzado**")
    
    analyzer = PDFAnalyzer()
    
    # Sidebar para gestión de documentos
    st.sidebar.header("📁 Gestión de Documentos")
    
    # Upload múltiples PDFs
    uploaded_files = st.sidebar.file_uploader(
        "Subir PDFs", type='pdf', accept_multiple_files=True)
    
    categories = st.sidebar.multiselect(
        "Categorías", ['Salud Pública', 'Decolonial', 'Metodología Cualitativa', 
                     'Psicología Ambiental', 'HUC', 'Otros'])
    
    if uploaded_files:
        for file in uploaded_files:
            if file not in analyzer.documents:
                text = analyzer.extract_text(file)
                concepts = analyzer.extract_key_concepts(text)
                analyzer.documents[file.name] = {
                    'text': text,
                    'concepts': concepts,
                    'categories': [],
                    'notes': '',
                    'tags': []
                }
        
        # Lista de documentos cargados
        doc_names = list(analyzer.documents.keys())
        selected_doc = st.sidebar.selectbox("Seleccionar documento", doc_names)
        
        # Gestión de categorías y notas
        with st.sidebar.expander("📝 Categorizar"):
            col1, col2 = st.columns(2)
            with col1:
                new_cat = st.text_input("Nueva categoría")
                if st.button("Añadir categoría") and new_cat:
                    if new_cat not in categories:
                        categories.append(new_cat)
            
            with col2:
                doc_categories = st.multiselect(
                    "Categorías del doc", categories,
                    default=analyzer.documents[selected_doc]['categories'])
                analyzer.documents[selected_doc]['categories'] = doc_categories
            
            notes = st.text_area("Notas", 
                               analyzer.documents[selected_doc]['notes'],
                               height=100)
            analyzer.documents[selected_doc]['notes'] = notes
        
        # Pestañas principales
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Análisis", "🗺️ Mapa Conceptual", 
                                        "🏷️ Categorías", "🔍 Cruzado"])
        
        with tab1:
            st.subheader(f"Análisis: {selected_doc}")
            
            concepts = analyzer.documents[selected_doc]['concepts']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### **Conceptos Clave (TF-IDF)**")
                tfidf_df = pd.DataFrame(concepts['tfidf'], 
                                      columns=['Concepto', 'Peso'])
                st.dataframe(tfidf_df.head(10))
            
            with col2:
                st.markdown("### **Entidades Nombradas**")
                entities_df = pd.DataFrame(concepts['entities'], 
                                         columns=['Entidad', 'Frecuencia'])
                st.dataframe(entities_df.head(10))
        
        with tab2:
            st.subheader("Mapa Conceptual Interactivo")
            
            # Construir y visualizar grafo
            all_concepts = concepts['tfidf'] + [(e[0], e[1]*10) for e in concepts['entities']]
            G = analyzer.build_concept_map(all_concepts)
            
            # Plotly network graph
            pos = nx.spring_layout(G, k=1, iterations=50)
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='gray'),
                                  hoverinfo='none', mode='lines')
            
            node_x, node_y, node_text, node_size = [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_size.append(G.nodes[node].get('size', 10)*5)
                node_text.append(f"{node}<br>Tamaño: {G.nodes[node].get('size', 0):.1f}")
            
            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                  textposition="middle center",
                                  hovertext=node_text,
                                  hoverinfo='text',
                                  marker=dict(size=node_size, color='LightSkyBlue',
                                            line=dict(width=1, color='white')))
            
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(showlegend=False, hovermode='closest',
                                         margin=dict(b=20,l=5,r=5,t=40),
                                         annotations=[ dict(text="Conceptos relacionados por co-ocurrencia",
                                                          showarrow=False, xref="paper", yref="paper",
                                                          x=0.005, y=-0.002) ],
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Gestión de Categorías")
            
            # Tabla resumen por categorías
            cat_summary = defaultdict(list)
            for doc_name, data in analyzer.documents.items():
                for cat in data['categories']:
                    cat_summary[cat].append(doc_name)
            
            for category, docs in cat_summary.items():
                with st.expander(f"📂 {category} ({len(docs)} docs)"):
                    for doc in docs:
                        st.write(f"- {doc}")
                        st.caption(analyzer.documents[doc]['notes'][:100] + "...")
        
        with tab4:
            st.subheader("Análisis Cruzado")
            if len(analyzer.documents) > 1:
                # Comparación entre documentos
                selected_compare = st.selectbox("Comparar con", 
                                              [d for d in doc_names if d != selected_doc])
                
                # Conceptos compartidos
                doc1_concepts = set([c[0] for c in analyzer.documents[selected_doc]['concepts']['tfidf'][:10]])
                doc2_concepts = set([c[0] for c in analyzer.documents[selected_compare]['concepts']['tfidf'][:10]])
                shared = doc1_concepts.intersection(doc2_concepts)
                
                st.metric("Conceptos compartidos", len(shared))
                if shared:
                    st.write("**Conceptos comunes:**", ", ".join(list(shared)[:5]))
    
    # Instrucciones
    with st.sidebar.expander("ℹ️ Instrucciones"):
        st.markdown("""
        1. **Sube PDFs** en la barra lateral
        2. **Categoriza** cada documento
        3. **Añade notas** relevantes
        4. **Explora** mapas conceptuales interactivos
        5. **Compara** múltiples documentos
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("💻 Para desarrollo local:\n`pip install streamlit PyPDF2 spacy networkx scikit-learn plotly matplotlib`\n`python -m spacy download es_core_news_sm`")

if __name__ == "__main__":
    main()
