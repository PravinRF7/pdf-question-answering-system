# PDF Question Answering System

## Description

The **PDF Question Answering System** is a web application that allows users to ask questions about the content of a PDF document. Built using Streamlit, this application leverages advanced language models and vector databases to provide accurate and contextually relevant answers based on the content of the provided PDF.

## Features

- **PDF Document Processing**: Extracts and processes text from PDF documents.
- **Text Chunking**: Splits extracted text into manageable chunks for efficient querying.
- **Vector Database**: Uses Chroma vector store to index text chunks for fast retrieval.
- **Query Enhancement**: Generates multiple variations of user queries to improve search accuracy.
- **Streamlit Interface**: Provides an easy-to-use web interface for interacting with the system.

## Installation

### Prerequisites

- Python 3.7 or later
- Pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/PravinRF7/pdf-question-answering-system.git
cd pdf-question-answering-system
