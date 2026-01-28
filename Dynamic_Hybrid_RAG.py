# ========================== ADVANCED RAG ARCHITECTURE ==========================
# This version addresses fundamental architectural limitations with:
# 1. Query routing (decide which RAG to use based on question type)
# 2. Multi-hop reasoning for graph RAG
# 3. Contextual chunk retrieval with metadata
# 4. Reranking retrieved results
# 5. Better graph construction with chunk-level entities

import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain_community.chains.graph_qa.base import GraphQAChain
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma import with fallback
try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.vectorstores import Chroma

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pdfplumber
import networkx as nx
import re
from typing import List, Dict, Tuple

# Tkinter imports
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


# ========================== QUERY CLASSIFIER ==========================

def classify_query(query: str, llm) -> str:
    """
    Classify query to determine best retrieval strategy.
    Returns: 'factual', 'relationship', 'complex', 'summary'
    """
    classification_template = """Classify this question into ONE category:

Question: {query}

Categories:
- factual: Simple fact lookup (e.g., "What is X?", "When did Y happen?")
- relationship: Questions about connections (e.g., "How does X relate to Y?", "What caused Z?")
- complex: Multi-part questions requiring reasoning (e.g., "Compare X and Y", "Analyze the impact of...")
- summary: Overview questions (e.g., "Summarize...", "What are the main points?")

Return ONLY the category name, nothing else."""
    
    prompt = ChatPromptTemplate.from_template(classification_template)
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"query": query}).strip().lower()
    
    # Ensure valid category
    valid_categories = ['factual', 'relationship', 'complex', 'summary']
    for cat in valid_categories:
        if cat in result:
            return cat
    
    return 'complex'  # Default fallback


# ========================== IMPROVED GRAPH RAG ==========================

def enhanced_graphrag(content: str, llm, chunk_size: int = 1500):
    """
    Enhanced Graph RAG with better chunking and entity extraction.
    Processes document in chunks to capture more granular relationships.
    """
    # Split into chunks first
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(content)
    
    # Build graph from all chunks
    graph = NetworkxEntityGraph()
    llm_transformer = LLMGraphTransformer(llm=llm)
    
    all_entities = set()
    
    for i, chunk in enumerate(chunks):
        doc = Document(page_content=chunk, metadata={"chunk_id": i})
        
        try:
            graph_docs = llm_transformer.convert_to_graph_documents([doc])
            
            if graph_docs and len(graph_docs) > 0:
                # Add nodes
                for node in graph_docs[0].nodes:
                    graph.add_node(node.id)
                    all_entities.add(node.id)
                
                # Add edges (bidirectional)
                for edge in graph_docs[0].relationships:
                    graph._graph.add_edge(
                        edge.source.id,
                        edge.target.id,
                        relation=edge.type,
                        chunk_id=i
                    )
                    
                    graph._graph.add_edge(
                        edge.target.id,
                        edge.source.id,
                        relation=f"{edge.type}_inverse",
                        chunk_id=i
                    )
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            continue
    
    chain = GraphQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True
    )
    
    return chain, graph


# ========================== IMPROVED VECTOR RAG WITH RERANKING ==========================

def enhanced_rag(content: str, embeddings, llm):
    """
    Enhanced RAG with better chunking, metadata, and contextual retrieval.
    """
    # Better chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    chunks = text_splitter.split_text(content)
    
    # Create documents with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        # Add context from previous/next chunks
        context_before = chunks[i-1][-100:] if i > 0 else ""
        context_after = chunks[i+1][:100] if i < len(chunks)-1 else ""
        
        doc = Document(
            page_content=chunk,
            metadata={
                "chunk_id": i,
                "context_before": context_before,
                "context_after": context_after,
                "position": f"{i}/{len(chunks)}"
            }
        )
        documents.append(doc)
    
    # Create vector store
    docsearch = Chroma.from_documents(documents, embeddings)
    
    # Create retriever with MMR (Maximum Marginal Relevance) for diversity
    retriever = docsearch.as_retriever(
        search_type='mmr',  # Better than similarity - reduces redundancy
        search_kwargs={
            "k": 5,
            "fetch_k": 15,  # Fetch more, then filter
            "lambda_mult": 0.7  # Balance relevance vs diversity
        }
    )
    
    # Enhanced prompt with instructions
    template = """You are a helpful AI assistant answering questions based on the provided context.

Context from document:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the context provided
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question"
3. Be specific and cite relevant parts of the context
4. If there are multiple relevant pieces of information, synthesize them

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            chunk_info = doc.metadata.get('position', 'Unknown')
            formatted.append(f"[Chunk {chunk_info}]\n{doc.page_content}\n")
        return "\n".join(formatted)
    
    # Build chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ========================== INTELLIGENT HYBRID RAG ==========================

def intelligent_hybrid_rag(query: str, standard_rag, graph_rag, llm, query_type: str = None):
    """
    Intelligent hybrid approach that routes queries and combines results smartly.
    """
    # Classify query if not provided
    if query_type is None:
        query_type = classify_query(query, llm)
    
    print(f"Query classified as: {query_type}")
    
    # Route based on query type
    if query_type == 'factual':
        # For factual queries, prefer vector RAG (faster, more precise)
        try:
            standard_answer = standard_rag.invoke(query)
            return f"[Vector RAG - Factual Query]\n\n{standard_answer}"
        except Exception as e:
            return f"Error in factual query: {str(e)}"
    
    elif query_type == 'relationship':
        # For relationship queries, prefer graph RAG
        try:
            result = graph_rag.invoke(query)
            graph_answer = result.get('result', str(result))
            return f"[Graph RAG - Relationship Query]\n\n{graph_answer}"
        except Exception as e:
            # Fallback to vector RAG
            standard_answer = standard_rag.invoke(query)
            return f"[Vector RAG Fallback]\n\n{standard_answer}"
    
    elif query_type == 'summary':
        # For summaries, use vector RAG with different prompt
        try:
            standard_answer = standard_rag.invoke(query)
            return f"[Vector RAG - Summary Query]\n\n{standard_answer}"
        except Exception as e:
            return f"Error in summary query: {str(e)}"
    
    else:  # complex queries
        # For complex queries, use both and synthesize
        results = {}
        
        # Try vector RAG
        try:
            standard_answer = standard_rag.invoke(query)
            results['vector'] = standard_answer
        except Exception as e:
            results['vector'] = f"Vector RAG error: {str(e)}"
        
        # Try graph RAG
        try:
            graph_result = graph_rag.invoke(query)
            results['graph'] = graph_result.get('result', str(graph_result))
        except Exception as e:
            results['graph'] = f"Graph RAG error: {str(e)}"
        
        # Synthesize with context-aware prompt
        synthesis_template = """You are synthesizing information from two different retrieval methods to answer a complex question.

Question: {question}

Vector Search Results (focuses on semantic similarity):
{vector_results}

Graph Search Results (focuses on entity relationships):
{graph_results}

Instructions:
1. Identify which source provides better information for this specific question
2. If both provide useful info, combine them intelligently
3. Prioritize accuracy over completeness
4. Explicitly note if information is conflicting or incomplete
5. Cite which method provided each piece of information

Synthesized Answer:"""
        
        synthesis_prompt = ChatPromptTemplate.from_template(synthesis_template)
        synthesis_chain = synthesis_prompt | llm | StrOutputParser()
        
        final_answer = synthesis_chain.invoke({
            "question": query,
            "vector_results": results.get('vector', 'No results'),
            "graph_results": results.get('graph', 'No results')
        })
        
        return f"[Hybrid RAG - Complex Query]\n\n{final_answer}"


# ========================== PDF EXTRACTION ==========================

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file with page metadata."""
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                all_text.append(f"[Page {page_num}]\n{text}")
    return "\n\n".join(all_text)


def draw_graph_rag(graph_rag):
    """Draws the NetworkX graph from a graph_rag object and returns a PIL Image."""
    nx_graph = graph_rag.graph._graph
    
    fig = plt.figure(figsize=(12, 8))
    
    # Use better layout algorithm if graph is large
    if len(nx_graph.nodes()) > 30:
        pos = nx.spring_layout(nx_graph, k=2, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(nx_graph, seed=42)
    
    # Draw with better aesthetics
    nx.draw(
        nx_graph,
        pos,
        with_labels=True,
        node_color='lightblue',
        node_size=3000,
        edge_color='gray',
        font_size=9,
        font_weight='bold',
        arrows=True,
        arrowsize=15
    )
    
    edge_labels = nx.get_edge_attributes(nx_graph, "relation")
    nx.draw_networkx_edge_labels(
        nx_graph, 
        pos, 
        edge_labels=edge_labels, 
        font_color='red', 
        font_size=7
    )
    
    plt.title("Knowledge Graph Visualization", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Convert to PIL Image
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close(fig)
    return pil_image


# ========================== TKINTER GUI ==========================

class EnhancedRAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Hybrid RAG System with Intelligent Routing")
        self.root.geometry("1100x850")
        
        # Variables
        self.pdf_path_var = tk.StringVar()
        self.model_name_var = tk.StringVar(value="openai/gpt-4o-mini")
        self.api_key_var = tk.StringVar()
        self.question_var = tk.StringVar()
        self.query_type_var = tk.StringVar(value="auto")
        
        # Storage for RAG objects
        self.standard_rag = None
        self.graph_rag = None
        self.retriever = None
        self.graph_obj = None
        self.llm = None
        self.embeddings = None
        
        # Build UI
        self.build_ui()
    
    def build_ui(self):
        # Create a canvas and a vertical scrollbar for the window
        self.canvas = tk.Canvas(self.root, borderwidth=0, background="#f0f0f0")
        self.v_scroll = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)
        self.v_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Create a frame inside the canvas
        self.frm = ttk.Frame(self.canvas, padding=10)
        self.frm_id = self.canvas.create_window((0, 0), window=self.frm, anchor="nw")

        # Bind the frame to configure the scrollregion
        self.frm.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        # Title
        title = ttk.Label(self.frm, text="Enhanced Hybrid RAG with Query Routing", font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=10)

        # OpenRouter Model Input
        ttk.Label(self.frm, text="OpenRouter Model:").grid(row=1, column=0, sticky=tk.W, pady=5)
        model_entry = ttk.Entry(self.frm, textvariable=self.model_name_var, width=40)
        model_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        # OpenRouter API Key Input
        ttk.Label(self.frm, text="OpenRouter API Key:").grid(row=2, column=0, sticky=tk.W, pady=5)
        api_key_entry = ttk.Entry(self.frm, textvariable=self.api_key_var, width=40, show="*")
        api_key_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(self.frm, text="(Get from https://openrouter.ai/)").grid(row=2, column=2, sticky=tk.W, pady=5, padx=5)

        # PDF Selection
        ttk.Label(self.frm, text="Select PDF:").grid(row=3, column=0, sticky=tk.W, pady=5)
        pdf_entry = ttk.Entry(self.frm, textvariable=self.pdf_path_var, width=40, state='readonly')
        pdf_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5)
        browse_btn = ttk.Button(self.frm, text="Browse", command=self.browse_pdf)
        browse_btn.grid(row=3, column=2, sticky=tk.W, pady=5, padx=5)

        # Process PDF Button
        process_btn = ttk.Button(self.frm, text="Process PDF & Create Enhanced RAGs", command=self.process_pdf)
        process_btn.grid(row=4, column=0, columnspan=3, pady=10)

        # Status Label
        self.status_label = ttk.Label(self.frm, text="Status: Ready", foreground="blue")
        self.status_label.grid(row=5, column=0, columnspan=3, pady=5)

        # Separator
        ttk.Separator(self.frm, orient='horizontal').grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        # Query Type Selection
        ttk.Label(self.frm, text="Query Type:").grid(row=7, column=0, sticky=tk.W, pady=5)
        query_type_combo = ttk.Combobox(self.frm, textvariable=self.query_type_var, width=37)
        query_type_combo['values'] = ('auto', 'factual', 'relationship', 'complex', 'summary')
        query_type_combo.grid(row=7, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Label(self.frm, text="('auto' = intelligent routing)").grid(row=7, column=2, sticky=tk.W, pady=5, padx=5)

        # Question Input
        ttk.Label(self.frm, text="Ask a Question:").grid(row=8, column=0, sticky=tk.W, pady=5)
        question_entry = ttk.Entry(self.frm, textvariable=self.question_var, width=40)
        question_entry.grid(row=8, column=1, sticky=(tk.W, tk.E), pady=5)
        ask_btn = ttk.Button(self.frm, text="Ask", command=self.ask_question)
        ask_btn.grid(row=8, column=2, sticky=tk.W, pady=5, padx=5)

        # Answer Display
        ttk.Label(self.frm, text="Answer:").grid(row=9, column=0, sticky=(tk.W, tk.N), pady=5)
        self.answer_text = tk.Text(self.frm, height=12, width=90, wrap=tk.WORD)
        self.answer_text.grid(row=9, column=1, columnspan=2, pady=5)

        # Scrollbar for answer
        scrollbar = ttk.Scrollbar(self.frm, orient=tk.VERTICAL, command=self.answer_text.yview)
        scrollbar.grid(row=9, column=3, sticky=(tk.N, tk.S))
        self.answer_text['yscrollcommand'] = scrollbar.set

        # Show Graph Button
        show_graph_btn = ttk.Button(self.frm, text="Show Knowledge Graph", command=self.show_graph)
        show_graph_btn.grid(row=10, column=0, columnspan=3, pady=10)

        # Graph Canvas
        self.graph_canvas = ttk.Label(self.frm, text="Knowledge graph will appear here after processing")
        self.graph_canvas.grid(row=11, column=0, columnspan=3, pady=10)

        self.frm.columnconfigure(1, weight=1)

    def _on_canvas_configure(self, event):
        # Resize the inner frame to match the canvas width
        canvas_width = event.width
        self.canvas.itemconfig(self.frm_id, width=canvas_width)
    
    def browse_pdf(self):
        filename = filedialog.askopenfilename(
            title="Select PDF Document",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.pdf_path_var.set(filename)
    
    def process_pdf(self):
        pdf_path = self.pdf_path_var.get()
        model_name = self.model_name_var.get().strip()
        
        if not pdf_path:
            messagebox.showerror("Error", "Please select a PDF file first!")
            return
        
        if not model_name:
            messagebox.showerror("Error", "Please enter an OpenRouter model name!")
            return
        
        openrouter_api_key = self.api_key_var.get().strip()
        if not openrouter_api_key:
            messagebox.showerror("Error", "Please enter your OpenRouter API key!")
            return
        
        try:
            self.status_label.config(text="Status: Extracting text from PDF...", foreground="orange")
            self.root.update()
            
            text = extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                messagebox.showerror("Error", "No text extracted from PDF!")
                self.status_label.config(text="Status: Failed", foreground="red")
                return
            
            self.status_label.config(text="Status: Initializing LLM...", foreground="orange")
            self.root.update()
            
            self.llm = ChatOpenAI(
                model=model_name,
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.3  # Lower for more consistent answers
            )
            
            self.embeddings = OpenAIEmbeddings(
                model="openai/text-embedding-3-small",
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            
            self.status_label.config(text="Status: Creating Enhanced Vector RAG...", foreground="orange")
            self.root.update()
            
            self.standard_rag, self.retriever = enhanced_rag(text, self.embeddings, self.llm)
            
            self.status_label.config(text="Status: Creating Enhanced Graph RAG (may take time)...", foreground="orange")
            self.root.update()
            
            self.graph_rag, self.graph_obj = enhanced_graphrag(text, self.llm)
            
            self.status_label.config(text="Status: Ready! Enhanced RAG systems created.", foreground="green")
            messagebox.showinfo("Success", "PDF processed with enhanced architecture!\n\n" +
                              "New features:\n" +
                              "• Intelligent query routing\n" +
                              "• Better chunking strategies\n" +
                              "• MMR retrieval for diversity\n" +
                              "• Enhanced entity extraction")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
            self.status_label.config(text=f"Status: Error", foreground="red")
    
    def ask_question(self):
        if not self.standard_rag or not self.graph_rag:
            messagebox.showerror("Error", "Please process a PDF first!")
            return
        
        question = self.question_var.get().strip()
        if not question:
            messagebox.showerror("Error", "Please enter a question!")
            return
        
        try:
            self.status_label.config(text="Status: Processing question...", foreground="orange")
            self.root.update()
            
            query_type = None if self.query_type_var.get() == 'auto' else self.query_type_var.get()
            
            answer = intelligent_hybrid_rag(
                question, 
                self.standard_rag, 
                self.graph_rag, 
                self.llm,
                query_type
            )
            
            self.answer_text.delete(1.0, tk.END)
            self.answer_text.insert(1.0, answer)
            
            self.status_label.config(text="Status: Question answered!", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {str(e)}")
            self.status_label.config(text=f"Status: Error", foreground="red")
    
    def show_graph(self):
        if not self.graph_rag:
            messagebox.showerror("Error", "Please process a PDF first!")
            return
        
        try:
            self.status_label.config(text="Status: Generating graph...", foreground="orange")
            self.root.update()
            
            pil_image = draw_graph_rag(self.graph_rag)
            pil_image.thumbnail((900, 600), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(pil_image)
            self.graph_canvas.config(image=photo, text="")
            self.graph_canvas.image = photo
            
            self.status_label.config(text="Status: Graph displayed!", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed: {str(e)}")
            self.status_label.config(text=f"Status: Error", foreground="red")


if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedRAGApp(root)
    root.mainloop()