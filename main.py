"""
Improved RAG (Retrieval-Augmented Generation) Application
Uses Ollama and LlamaIndex for document retrieval and question answering

Features:
- Command-line arguments for customization
- Rich text formatting for better user experience
- Improved error handling and logging
- Citation support to track sources
- Stream mode option for real-time responses
- Post-processing with similarity threshold
- Automatic virtual environment activation
"""

import os
import sys
import time
import logging
import argparse
import subprocess
from typing import List, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint

# Verificação e ativação do ambiente virtual (venv)
def ensure_venv():
    """Verifica se está rodando em um ambiente virtual, se não, tenta ativar ou criar um."""
    # Verifica se já está em um ambiente virtual
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return True  # Já está em um ambiente virtual
    
    # Usar print padrão aqui porque o rico pode não estar disponível ainda
    print("Verificando ambiente virtual...")
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
    
    # Verifica se o ambiente virtual já existe
    if not os.path.exists(venv_dir):
        print("Ambiente virtual não encontrado. Criando...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
            print("Ambiente virtual criado com sucesso.")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao criar ambiente virtual: {e}")
            return False
    
    # Determina o caminho para o script de ativação baseado no sistema
    if sys.platform == "win32":
        activate_script = os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Linux/Mac
        activate_script = os.path.join(venv_dir, "bin", "python")
    
    if os.path.exists(activate_script):
        # Re-executa o script atual com o Python do ambiente virtual
        print("Ativando ambiente virtual...")
        os.execv(activate_script, [activate_script] + sys.argv)
    else:
        print(f"Não foi possível encontrar o script de ativação: {activate_script}")
        return False

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

# 🧠 Default Configurations
INDEX_DIR = "./meu_indice"
DOCUMENT_DIR = "my_docs"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
DEFAULT_LLM_MODEL = "phi3"
DEFAULT_EMBED_MODEL = "all-minilm"
OLLAMA_BASE_URL = "http://localhost:11434"
TOP_K = 3  # Number of documents to retrieve
SIMILARITY_CUTOFF = 0.7  # Minimum similarity score threshold (0.0 to 1.0)

# 🛠️ Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("rag_app.log"), logging.StreamHandler()]
)

# Enhanced console interface
console = Console()

# RAG prompt template
QA_TEMPLATE = PromptTemplate(
    """You are a helpful and accurate assistant who answers questions based solely on the provided information.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {query}
    
    INSTRUCTIONS:
    - Answer only based on the information in the CONTEXT above.
    - If the CONTEXT doesn't contain sufficient information, clearly indicate that you don't have enough information.
    - Don't make up information or use external knowledge.
    - Be clear and concise.
    - When appropriate, structure your response in a list format or with bullet points for clarity.
    
    ANSWER:"""
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RAG application using Ollama and LlamaIndex")
    parser.add_argument("--llm", default=DEFAULT_LLM_MODEL, help=f"LLM model to use (default: {DEFAULT_LLM_MODEL})")
    parser.add_argument("--embed", default=DEFAULT_EMBED_MODEL, help=f"Embedding model (default: {DEFAULT_EMBED_MODEL})")
    parser.add_argument("--docs", default=DOCUMENT_DIR, help=f"Documents directory (default: {DOCUMENT_DIR})")
    parser.add_argument("--base-url", default=OLLAMA_BASE_URL, help=f"Ollama base URL (default: {OLLAMA_BASE_URL})")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Number of documents to retrieve (default: {TOP_K})")
    parser.add_argument("--similarity", type=float, default=SIMILARITY_CUTOFF,
                      help=f"Minimum similarity score (default: {SIMILARITY_CUTOFF})")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuild of the index")
    parser.add_argument("--update-index", action="store_true", help="Update index with new documents only")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, 
                      help=f"Chunk size for document splitting (default: {CHUNK_SIZE})")
    return parser.parse_args()

def setup_models(args):
    """Setup LLM and Embedding models with proper configurations."""
    # Token counter for tracking usage
    token_counter = TokenCountingHandler(
        tokenizer=None,  # Uses default tokenizer
        verbose=args.stream  # Show token counts in stream mode
    )
    callback_manager = CallbackManager([token_counter])
    
    # LLM configuration
    llm = Ollama(
        model=args.llm,
        base_url=args.base_url,
        request_timeout=600,
        temperature=0.1,
        context_window=4096,
        stream=args.stream,
        callback_manager=callback_manager
    )

    # Embedding model configuration
    embed_model = OllamaEmbedding(
        model_name=args.embed,
        base_url=args.base_url,
        request_timeout=180
    )
    
    # Configure LlamaIndex global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.callback_manager = callback_manager
    
    return llm, embed_model, token_counter

def setup_index(args, embed_model):
    """Setup or load the vector index."""
    # Check if we have an existing index
    has_existing_index = os.path.exists(INDEX_DIR)
    
    # Load documents based on different scenarios
    if has_existing_index and not args.rebuild_index and not args.update_index:
        # Normal load - just use existing index
        with Progress(
            SpinnerColumn(),
            TextColumn("[yellow]🔁 Loading saved index...[/yellow]"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("", total=1)
            storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
            index = load_index_from_storage(storage_context, embed_model=embed_model)
            progress.update(task, advance=1)
        console.print("[green]✓ Index loaded successfully[/green]")
        return index
        
    # Load all documents from directory
    console.print("[blue]📚 Reading and processing documents...[/blue]")
    try:
        # Support for both PDF and markdown files
        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]Loading documents...[/blue]"),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("", total=1)
            docs = SimpleDirectoryReader(
                args.docs, 
                required_exts=[".pdf", ".md"]
            ).load_data()
            progress.update(task, advance=1)
        
        if not docs:
            console.print(f"[red]⚠️ No documents found in {args.docs}[/red]")
            return None
            
        console.print(f"[green]✓ Loaded {len(docs)} documents[/green]")
        
        # Create node parser
        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]Splitting documents into chunks...[/blue]"),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("", total=1)
            parser = SimpleNodeParser.from_defaults(
                chunk_size=args.chunk_size, 
                chunk_overlap=CHUNK_OVERLAP
            )
            nodes = parser.get_nodes_from_documents(docs)
            progress.update(task, advance=1)
        
        # Different index handling based on flags
        if args.update_index and has_existing_index:
            # Update existing index with new documents
            console.print("[yellow]🔄 Updating index with new documents...[/yellow]")
            
            # Load existing index
            with Progress(
                SpinnerColumn(),
                TextColumn("[yellow]Loading existing index...[/yellow]"),
                TimeElapsedColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("", total=1)
                storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
                index = load_index_from_storage(storage_context, embed_model=embed_model)
                progress.update(task, advance=1)
            
            # Get list of documents already in the index to avoid duplicates
            with Progress(
                SpinnerColumn(),
                TextColumn("[yellow]Checking existing documents...[/yellow]"),
                TimeElapsedColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("", total=1)
                existing_docs = set()
                for node in index.docstore.docs.values():
                    if hasattr(node, 'metadata') and 'file_name' in node.metadata:
                        existing_docs.add(node.metadata['file_name'])
                progress.update(task, advance=1)
            
            # Filter only new documents' nodes
            new_nodes = []
            for node in nodes:
                if 'file_name' in node.metadata and node.metadata['file_name'] not in existing_docs:
                    new_nodes.append(node)
            
            # If no new documents were found
            if not new_nodes:
                console.print("[yellow]ℹ️ No new documents found to add to the index.[/yellow]")
                return index
                
            console.print(f"[green]✓ Found {len(new_nodes)} new documents to add[/green]")
            
            # Insert new documents into the index
            with Progress(
                SpinnerColumn(),
                TextColumn("[green]Adding new documents to index...[/green]"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("", total=len(new_nodes))
                for node in new_nodes:
                    index.insert(node)
                    progress.update(task, advance=1)
                
            # Save updated index
            with Progress(
                SpinnerColumn(),
                TextColumn("[green]Saving updated index...[/green]"),
                TimeElapsedColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("", total=1)
                index.storage_context.persist(persist_dir=INDEX_DIR)
                progress.update(task, advance=1)
                
            console.print("[green]✅ Index successfully updated.[/green]")
            
        else:
            # Create new index (either first time or forced rebuild)
            msg = "🔄 Rebuilding" if has_existing_index else "🧠 Creating new"
            console.print(f"[yellow]{msg} vector index...[/yellow]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[yellow]Creating vector index...[/yellow]"),
                TimeElapsedColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("", total=1)
                index = VectorStoreIndex(nodes, embed_model=embed_model)
                progress.update(task, advance=1)
            
            # Save index
            with Progress(
                SpinnerColumn(),
                TextColumn("[yellow]Saving index to disk...[/yellow]"),
                TimeElapsedColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("", total=1)
                index.storage_context.persist(persist_dir=INDEX_DIR)
                progress.update(task, advance=1)
                
            console.print("[green]✅ Index successfully saved.[/green]")
            
    except Exception as e:
        console.print(f"[red]❌ Error creating index: {str(e)}[/red]")
        logging.error(f"Error creating index: {e}")
        return None
        
    return index

def setup_retriever(index, args):
    """Setup retriever with similarity threshold."""
    try:
        # Configure retriever with similarity score threshold
        retriever = index.as_retriever(
            similarity_top_k=args.top_k
        )
        
        # Add post-processing for similarity cutoff
        if args.similarity > 0:
            retriever.node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=args.similarity)]
            
        return retriever
    except Exception as e:
        console.print(f"[red]❌ Error configuring retriever: {str(e)}[/red]")
        logging.error(f"Error configuring retriever: {e}")
        return None

def format_sources(nodes):
    """Format source information for display."""
    sources = []
    for i, node in enumerate(nodes):
        doc_source = node.metadata.get("file_name", "Unknown")
        # Try to get page number if available
        page_info = f", page {node.metadata.get('page_label')}" if "page_label" in node.metadata else ""
        sources.append(f"[{i+1}] {doc_source}{page_info}")
    
    return sources

def process_query(query, retriever, llm, args):
    """Process a user query and return the answer with metadata."""
    try:
        start = time.time()
        
        # Retrieve relevant documents
        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]Searching for relevant documents...[/blue]"),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("", total=1)
            nodes = retriever.retrieve(query)
            progress.update(task, advance=1)
        
        # Check if we found any relevant documents
        if not nodes:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "time": time.time() - start
            }
        
        # Extract text and format for context
        context_text = "\n\n".join([n.get_content() for n in nodes])
        
        # Use the template for better prompting
        prompt = QA_TEMPLATE.format(
            context=context_text,
            query=query
        )
        
        # Generate response
        if args.stream:
            # Stream mode requires special handling
            console.print("\n[cyan]Chatbot:[/cyan]", end=" ")
            collected_response = []
            
            # Stream chunks as they come
            for chunk in llm.stream_complete(prompt):
                chunk_text = chunk.delta
                console.print(chunk_text, end="")
                collected_response.append(chunk_text)
            
            response_text = "".join(collected_response)
            console.print()  # Add a newline after streaming
        else:
            # Non-streaming mode
            console.print("[cyan]Generating response...[/cyan]")
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Thinking about the best answer...[/cyan]"),
                TimeElapsedColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("", total=None)  # Indeterminate progress
                response = llm.complete(prompt)
                progress.update(task, completed=True)
            response_text = response.text
        
        end = time.time()
        
        # Prepare source information
        sources = format_sources(nodes)
        
        return {
            "answer": response_text.strip(),
            "sources": sources,
            "time": end - start
        }
        
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        return {
            "answer": f"❌ Error processing your question: {str(e)}",
            "sources": [],
            "time": 0
        }

def print_welcome():
    """Print welcome message and instructions."""
    console.print("[bold blue]🤖 RAG Chat - Document Query System[/bold blue]")
    console.print("[yellow]Enter your questions or 'exit' to quit.[/yellow]")
    console.print("[yellow]Special commands:[/yellow]")
    console.print("  [green]!help[/green] - Show this message")
    console.print("  [green]!clear[/green] - Clear the screen")
    console.print("  [green]!info[/green] - Display current configuration")
    console.print("  [green]!update[/green] - Update index with new documents")
    console.print("  [green]!rebuild[/green] - Rebuild the complete index")
    console.print()

def main():
    """Main function to run the RAG application."""
    # Ativar o ambiente virtual antes de mais nada
    if not ensure_venv():
        print("[red]Falha ao garantir o ambiente virtual. Saindo...[/red]")
        return
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup models
    llm, embed_model, token_counter = setup_models(args)
    
    # Setup index
    index = setup_index(args, embed_model)
    if not index:
        console.print("[red]❌ Failed to set up the index. Exiting.[/red]")
        return
    
    # Setup retriever
    retriever = setup_retriever(index, args)
    if not retriever:
        console.print("[red]❌ Failed to set up the retriever. Exiting.[/red]")
        return
    
    # Print welcome message
    print_welcome()
    
    # Main interaction loop
    while True:
        try:
            # Get user query
            query = console.input("[bold green]You:[/bold green] ")
            
            # Handle special commands
            if query.strip().lower() in ["exit", "quit"]:
                console.print("[yellow]Ending chat. See you soon![/yellow]")
                break
                
            elif query.strip() == "!help":
                print_welcome()
                continue
                
            elif query.strip() == "!clear":
                console.clear()
                continue
                
            elif query.strip() == "!info":
                console.print(f"[blue]LLM Model:[/blue] {args.llm}")
                console.print(f"[blue]Embed Model:[/blue] {args.embed}")
                console.print(f"[blue]Documents:[/blue] {args.docs}")
                console.print(f"[blue]Top K:[/blue] {args.top_k}")
                console.print(f"[blue]Similarity:[/blue] {args.similarity}")
                console.print(f"[blue]Streaming:[/blue] {'Enabled' if args.stream else 'Disabled'}")
                continue
                
            elif query.strip() == "!update":
                console.print("[yellow]🔄 Updating index with new documents...[/yellow]")
                # Temporarily enable update flag
                args.update_index = True
                
                # Re-setup index and retriever
                index = setup_index(args, embed_model)
                if index:
                    retriever = setup_retriever(index, args)
                    console.print("[green]✅ Index successfully updated.[/green]")
                else:
                    console.print("[red]❌ Failed to update the index.[/red]")
                
                # Reset flag
                args.update_index = False
                continue
                
            elif query.strip() == "!rebuild":
                console.print("[yellow]🔄 Rebuilding complete index...[/yellow]")
                # Temporarily enable rebuild flag
                args.rebuild_index = True
                
                # Re-setup index and retriever
                index = setup_index(args, embed_model)
                if index:
                    retriever = setup_retriever(index, args)
                    console.print("[green]✅ Index successfully rebuilt.[/green]")
                else:
                    console.print("[red]❌ Failed to rebuild the index.[/red]")
                
                # Reset flag
                args.rebuild_index = False
                continue
            
            # Skip empty queries
            if not query.strip():
                continue
            
            # Process query
            result = process_query(query, retriever, llm, args)
            
            # Skip printing response if streaming (already printed during streaming)
            if not args.stream:
                # Display the answer
                console.print("\n[cyan]Chatbot:[/cyan]", end=" ")
                console.print(Markdown(result["answer"]))
            
            # Display sources if available
            if result["sources"]:
                console.print("\n[yellow]Sources:[/yellow]")
                for source in result["sources"]:
                    console.print(f"  {source}")
            
            # Display response time
            console.print(f"\n[dim]⏱️ Response time: {round(result['time'], 2)}s[/dim]")
            
            # Display token usage if available
            if token_counter.total_llm_token_count > 0:
                console.print(f"[dim]🔢 Tokens: {token_counter.total_llm_token_count} (input: {token_counter.prompt_llm_token_count}, output: {token_counter.completion_llm_token_count})[/dim]")
            
            console.print() # Add an empty line for readability
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user. Type 'exit' to quit.[/yellow]")
            continue
            
        except Exception as e:
            console.print(f"\n[red]❌ Error processing question: {str(e)}[/red]")
            logging.error(f"Error processing question: {e}")
            continue

if __name__ == "__main__":
    # Garantir que estamos em um ambiente virtual antes de continuar
    ensure_venv()
    
    # Verificar se as dependências estão instaladas
    try:
        import rich
        import llama_index
    except ImportError:
        # Use print comum aqui porque rich pode não estar disponível
        print("Instalando dependências necessárias...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("Dependências instaladas com sucesso.")
            # Não precisamos recarregar aqui, pois o script já será reiniciado
            # dentro do ambiente virtual
        except Exception as e:
            print(f"Erro ao instalar dependências: {e}")
            sys.exit(1)
    
    # Continue com a execução normal do programa
    main()
