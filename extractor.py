import os
import json
import csv
import logging
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, AnalyzeDocumentRequest

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set your Azure endpoint and key
endpoint = os.getenv("AZURE_ENDPOINT")
key = os.getenv("AZURE_KEY")

# Helper functions
def get_words(page, line):
    result = []
    for word in page.words:
        if _in_span(word, line.spans):
            result.append(word)
    return result

def _in_span(word, spans):
    for span in spans:
        if word.span.offset >= span.offset and (
            word.span.offset + word.span.length
        ) <= (span.offset + span.length):
            return True
    return False

def analyze_layout(formUrl: str, output_json_path: str) -> dict:
    if os.path.exists(output_json_path):
        logging.info("JSON file already exists. Loading the existing result.")
        with open(output_json_path, "r") as json_file:
            return json.load(json_file)

    try:
        with open(formUrl, "rb") as file:
            file_content = file.read()

        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", AnalyzeDocumentRequest(bytes_source=file_content)
        )

        result: AnalyzeResult = poller.result()
        result_dict = result.as_dict()

        with open(output_json_path, "w") as json_file:
            json.dump(result_dict, json_file, indent=4)
        logging.info(f"Analysis result saved to {output_json_path}")
        return result_dict

    except Exception as e:
        logging.error(f"Error analyzing document: {e}")
        raise

def extract_and_format_tables(data: dict, output_folder: str, table_offset: int = 0) -> int:
    if "tables" not in data:
        logging.warning("No tables found in the JSON data.")
        return table_offset

    os.makedirs(output_folder, exist_ok=True)

    for table_index, table in enumerate(data["tables"]):
        row_count = table["rowCount"]
        column_count = table["columnCount"]
        cells = table["cells"]

        table_data = [["" for _ in range(column_count)] for _ in range(row_count)]

        for cell in cells:
            row_index = cell["rowIndex"]
            column_index = cell["columnIndex"]
            content = cell["content"]
            table_data[row_index][column_index] = content

        csv_file_path = os.path.join(output_folder, f"table_{table_index + 1 + table_offset}.csv")
        with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(table_data)
        logging.info(f"Table {table_index + 1 + table_offset} saved to {csv_file_path}")

    return table_offset + len(data["tables"])

def extract_and_save_lines(data: dict, output_folder: str, line_offset: int = 0) -> int:
    if "pages" not in data:
        logging.warning("No pages found in the JSON data.")
        return line_offset

    os.makedirs(output_folder, exist_ok=True)

    for page_index, page in enumerate(data["pages"]):
        lines = page.get("lines", [])
        page_text = "\n".join(line["content"] for line in lines)

        txt_file_path = os.path.join(output_folder, f"page_{page_index + 1 + line_offset}.txt")
        with open(txt_file_path, "w", encoding="utf-8") as txtfile:
            txtfile.write(page_text)
        logging.info(f"Page {page_index + 1 + line_offset} lines saved to {txt_file_path}")

    return line_offset + len(data["pages"])

def main():
    input_folder = r"./KnowledgeBase"
    output_json_folder = r"./TRIAL_DOC_results"
    tables_output_folder = r"./ExtractedTables"
    lines_output_folder = r"./ExtractedLines"

    os.makedirs(output_json_folder, exist_ok=True)
    os.makedirs(tables_output_folder, exist_ok=True)
    os.makedirs(lines_output_folder, exist_ok=True)

    table_offset = 0
    line_offset = 0

    for pdf_file in os.listdir(input_folder):
        if pdf_file.endswith(".pdf"):
            formUrl = os.path.join(input_folder, pdf_file)
            output_json_path = os.path.join(output_json_folder, f"{pdf_file}_result.json")

            # Analyze document layout and save result as JSON
            result_dict = analyze_layout(formUrl, output_json_path)

            # Extract and save tables as CSV files
            table_offset = extract_and_format_tables(result_dict, tables_output_folder, table_offset)

            # Extract and save lines as TXT files
            line_offset = extract_and_save_lines(result_dict, lines_output_folder, line_offset)

if __name__ == "__main__":
    main()