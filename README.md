# Multi-method RAG Bot

This repository contains a Multi-method Retrieval-Augmented Generation (RAG) bot, designed to use various LLM and LVLM and frameworks for improved results. The bot is built with Streamlit, making it easy to interact with via a web interface.

## Features
- **Multi-method RAG:** Combines different methods for retrieval-augmented generation.
- **Ollama LLaMA3 integration:** Utilizes the LLaMA3 model through Ollama via Langchain for the core LLM and KG functionality.
- **LVLM GPT-4o support:** Integrates GPT-4o, to integrate multimedia RAG capabilities.
- **Extensible:** Add new models and frameworks by modifying the backend logic.

## Prerequisites
Before running the bot, ensure you have the following installed:

- Python 3.8 or above
- Required packages in `requirements.txt`
- [Ollama](https://ollama.com/) (to run LLaMA3)
- [Langchain](https://python.langchain.com/) (for GPT-4o and LLaMA3)

## Installation

1. Clone the repository
    
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install **Ollama** and **Langchain** as dependencies:
    - Follow the installation instructions for [Ollama](https://ollama.com) to set up LLaMA3.
    - Install Langchain using pip:
      ```bash
      pip install langchain
      ```

4. Set up the GPT-4o API key:
    - Create a `.env` file in the toolbox directory.
    - Add the following line to the `.env` file:
      ```env
      AZURE_OPENAI_API_KEY=your_api_key_here
      AZURE_OPENAI_ENDPOINT=your_link_here
      ```
      Replace `your_api_key_here`and  `your_link_here` with your actual GPT-4o API key and link.
    
    Note: You must manually generate the GPT-4o API key and store it in the `.env` file.

## Running the Bot

To launch the Multi-method RAG bot, use Streamlit:

```bash
streamlit run runbot.py