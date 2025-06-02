# RAG Application with Ollama and LlamaIndex

This application provides a Retrieval-Augmented Generation (RAG) system using Ollama for local language model inference and LlamaIndex for document management and retrieval.

## Features

- **Document Indexing**: Automatically creates and manages vector indices of your documents
- **Semantic Search**: Retrieves relevant document chunks based on your queries
- **Rich Text Interface**: Improved terminal interface with formatting and colors
- **Streaming Support**: Optional streaming mode for real-time responses
- **Citation Support**: Shows document sources for responses
- **Configurable**: Command-line options to customize behavior
- **Token Counting**: Tracks token usage for input and output
- **Automatic Virtual Environment**: Creates and activates a virtual environment automatically

## Requirements

- Python 3.8+ (on Ubuntu, use `python3`)
- Ollama running locally (or remotely with URL configuration)
- PDFs and/or Markdown documents in the `my_docs` folder
- `python3-venv` package (usually pre-installed on most Linux distributions)

## Installation and Setup

### First-time setup on a new computer

1. Clone the repository:
   ```bash
   git clone <your-repository-url> ollama-rag
   cd ollama-rag
   ```

2. Run the application:
   ```bash
   python3 main.py
   ```

   The script will automatically:
   - Create a virtual environment in a `venv` directory
   - Install all required dependencies from the `requirements.txt` file
   - Activate the environment and run the application

### Running the application after setup

Simply run:
```bash
python3 main.py
```

The script will detect the existing virtual environment and automatically activate it.

### Manual installation (if automatic setup fails)

If you prefer to set up manually:

1. Create and activate a virtual environment:
   ```bash
   # On Linux/macOS:
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows:
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Usage

Basic usage:

```bash
python3 main.py
```

With options:

```bash
python3 main.py --llm phi3 --stream --top-k 5
```

### Command-line Arguments

- `--llm MODEL`: LLM model to use from Ollama (default: "phi3")
- `--embed MODEL`: Embedding model to use (default: "all-minilm")
- `--docs DIR`: Documents directory (default: "my_docs")
- `--base-url URL`: Ollama base URL (default: "http://localhost:11434")
- `--stream`: Enable streaming mode for real-time responses
- `--top-k N`: Number of documents to retrieve (default: 3)
- `--similarity FLOAT`: Minimum similarity score threshold (default: 0.7)
- `--rebuild-index`: Force rebuild of the document index
- `--update-index`: Update index with only new documents (no rebuilding)
- `--chunk-size N`: Chunk size for document splitting (default: 512)

### Chat Commands

While using the application, you can use these special commands:

- `!help`: Display help information
- `!clear`: Clear the screen
- `!info`: Show current configuration
- `!update`: Update index with new documents without rebuilding
- `!rebuild`: Completely rebuild the document index
- `exit`, `quit`: Exit the application

## Troubleshooting

### Common issues:

1. **Missing Python**: If you see `Command 'python' not found`, use `python3` instead (common on Ubuntu)
2. **Missing virtual environment package**: If you get errors creating the virtual environment:
   ```bash
   sudo apt-get update && sudo apt-get install -y python3-venv
   ```
3. **Ollama not running**: If you see connection errors, make sure Ollama is running:
   ```bash
   ollama serve
   ```
4. **Missing models**: If models aren't found, download them with:
   ```bash
   ollama pull phi3
   ollama pull all-minilm
   ```

## Improvements

1. **Better User Interface**: Rich text formatting for clarity
2. **Command Line Arguments**: Easy customization without code changes
3. **Better Error Handling**: Improved logging and error reporting
4. **Streaming Support**: Option for real-time response generation
5. **Improved Prompt Template**: Better instructions for the LLM
6. **Source Citations**: Track where information comes from
7. **Post-Processing**: Filter out irrelevant context with similarity threshold
8. **Token Counting**: Track token usage for optimization
9. **Special Commands**: Interactive commands during chat session
10. **Code Organization**: Better function structure and documentation
11. **Automatic Virtual Environment**: Seamless environment management

## Example

```
🤖 RAG Chat - Document Query System
Enter your questions or 'exit' to quit.
Special commands:
  !help - Show this message
  !clear - Clear the screen
  !info - Display current configuration
  !update - Update index with new documents
  !rebuild - Rebuild the complete index

You: What is a singleton pattern?

Chatbot: Based on the provided context, the Singleton pattern is a design pattern that ensures a class has only one instance and provides a global point of access to that instance. In Delphi, the Singleton pattern is implemented by creating a class with a private constructor and a static class method that returns the single instance of the class.

Sources:
  [1] delphi_coding_standards.pdf

⏱️ Response time: 1.45s
```

## Known Limitations

- Only works with PDF and Markdown files
- Requires Ollama running locally or accessible via network
- Index storage is fixed to the `./meu_indice` directory
