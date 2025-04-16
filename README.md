# RAG-Based Website Navigation (SIH 2024)

FastAPI backend implementing a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions based on local text content (`page_content.txt`). Features automatic language detection and translation between English and Hindi.

## Technology Stack & Features

* **Framework:** FastAPI (with streaming responses)
* **RAG Core:** LangChain
* **LLM:** Ollama (`phi3.5:3.8b-mini-instruct-q8_0`)
* **Embeddings:** Ollama (`nomic-embed-text`)
* **Vector Store:** ChromaDB
* **Translation:** `argostranslate` (Hindi <-> English), `langdetect`
* **Knowledge Base:** Sourced from `page_content.txt`

