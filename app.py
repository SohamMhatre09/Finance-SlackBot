# app.py
from flask import Flask, request, jsonify
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
import os
import re
import logging
from typing import Dict, List
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv  # Import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Slack app setup
slack_app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),  # Use environment variable
    signing_secret=os.getenv("SLACK_SIGNING_SECRET")  # Use environment variable
)
handler = SlackRequestHandler(slack_app)

# Configure logging
logging.basicConfig(level=logging.INFO)

class LangChainRAG:
    def __init__(
        self, 
        tables_dir: str, 
        text_dir: str, 
        gemini_api_key: str = None
    ):
        self.tables_dir = tables_dir
        self.text_dir = text_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=gemini_api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        self.retriever = None

    def clean_data(self, value: str) -> str:
        """Clean text values while preserving special characters like %, cr, etc."""
        if isinstance(value, (int, float)):
            return str(value)
        # Preserve special characters like %, cr, and other currency symbols
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,$€£%-]', ' ', str(value))
        return ' '.join(cleaned.split())

    def process_and_load_documents(self):
        """Process and load documents with proper metadata."""
        documents = []

        # Process text files
        for file in os.listdir(self.text_dir):
            if file.endswith('.txt'):
                file_path = os.path.join(self.text_dir, file)
                try:
                    # Try UTF-8 first
                    try:
                        loader = TextLoader(file_path, encoding='utf-8')
                        text_docs = loader.load()
                    except UnicodeDecodeError:
                        # Fallback to cp1252 if UTF-8 fails
                        logging.warning(f"UTF-8 failed for {file}, trying cp1252")
                        loader = TextLoader(file_path, encoding='cp1252')
                        text_docs = loader.load()

                    for doc in text_docs:
                        doc.metadata.update({
                            'doc_type': 'text',
                            'source': file
                        })
                    documents.extend(text_docs)
                except Exception as e:
                    logging.error(f"Error loading {file}: {e}")

        # Process CSV files
        for file in os.listdir(self.tables_dir):
            if file.endswith('.csv'):
                try:
                    loader = CSVLoader(os.path.join(self.tables_dir, file))
                    table_docs = loader.load()
                    table_name = f"table_{file.split('.')[0]}"
                    for doc in table_docs:
                        doc.metadata.update({
                            'doc_type': 'table',
                            'table_name': table_name
                        })
                    documents.extend(table_docs)
                except Exception as e:
                    logging.error(f"Error loading {file}: {e}")

        return documents

    def setup_rag_pipeline(self):
        """Set up the RAG pipeline with enhanced metadata handling."""
        documents = self.process_and_load_documents()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings
        )
        
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8}
        )

    def process_retrieved_docs(self, docs: List) -> Dict:
        """Process retrieved documents to find relevant tables and lines."""
        table_data = {}
        text_lines = []
        
        for doc in docs:
            if doc.metadata.get('doc_type') == 'table':
                table_name = doc.metadata.get('table_name')
                content = '\n'.join([self.clean_data(item) 
                                   for item in doc.page_content.split('\n')])
                if table_name not in table_data:
                    table_data[table_name] = []
                table_data[table_name].append(content)
            else:
                text_lines.append(self.clean_data(doc.page_content))

        # Select top 2 tables with most entries
        sorted_tables = sorted(table_data.items(), 
                             key=lambda x: len(x[1]), reverse=True)[:2]
        top_tables = {k: v[:3] for k, v in sorted_tables}  # Take top 3 entries per table
        
        return {
            'tables': top_tables,
            'text_lines': list(set(text_lines))[:5]  # Deduplicate and take top 5
        }

    def format_context(self, processed_data: Dict) -> str:
        """Format processed data into LLM-readable context."""
        context_str = "Relevant Information Extracted from Documents:\n\n"
        
        # Add tables section
        context_str += "=== Important Tables ===\n"
        for table_name, entries in processed_data['tables'].items():
            context_str += f"\n**{table_name}**\n"
            context_str += '\n'.join([f"- {entry}" for entry in entries])
            context_str += "\n"
        
        # Add text lines section
        context_str += "\n=== Relevant Text Lines ===\n"
        context_str += '\n'.join([f"- {line}" for line in processed_data['text_lines']])
        
        return context_str

    def query_document(self, question: str) -> Dict:
        """Query documents using OCR-aware RAG pipeline."""
        if not self.retriever:
            raise ValueError("Initialize RAG pipeline with setup_rag_pipeline() first")
            
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(question)
        
        # Process documents into structured data
        processed_data = self.process_retrieved_docs(docs)
        
        # Format context for LLM
        context = self.format_context(processed_data)
        
        # Create and execute prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You're an analyst working with OCR-processed documents. Use the 
             following context to answer. Tables may contain OCR errors - consider 
             approximate values and multiple data sources. Pay special attention to 
             numerical values with units like crores (cr) or percentages (%). Provide a 
             concise and natural response without mentioning table names or technical 
             details. Include a brief reasoning or explanation where appropriate.
             
             Context:
             {context}"""),
            ("human", "Question: {question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question})
        
        return {
            'answer': response,
            'context_used': processed_data
        }

# Initialize RAG system
rag_system = LangChainRAG(
    tables_dir="ExtractedTables",
    text_dir="ExtractedLines",
    gemini_api_key=os.getenv("GEMINI_API_KEY")  # Use environment variable
)

rag_system.setup_rag_pipeline()

# Flask endpoint for Slack
@app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

# Slack event listener
@slack_app.event("app_mention")
def handle_app_mention(event, say):
    question = event['text']
    result = rag_system.query_document(question)
    say(result['answer'])

# Run the Flask app
if __name__ == "__main__":
    app.run(port=3000)