import webbrowser
import requests
import json
import subprocess
from seleniumbase import BaseCase
from bs4 import BeautifulSoup
import os
from typing import Optional, Tuple, List, Union
from urllib.parse import urlparse
from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
)
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    KeywordExtractor
)
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition
)
from llama_index.extractors.marvin import MarvinMetadataExtractor
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel, Field
from typing import List

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1,max_tokens=1024)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-large", embed_batch_size=100
)
Settings.chunk_size = 512
Settings.chunk_overlap = 20



# Tool Function: open_browser
def open_browser(urls: Union[str, List[str]], use_chrome: bool = False, new_window: bool = False, new_tab: bool = False) -> str:
    """Open multiple URLs in the specified browser."""
    if not isinstance(urls, list):
        urls = [urls]  # Ensure urls is a list

    try:
        if use_chrome:
            browser = webbrowser.Chrome('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        else:
            browser = webbrowser.get()

        for url in urls:
            if new_window:
                browser.open_new(url)
            elif new_tab:
                browser.open_new_tab(url)
            else:
                browser.open(url)

        return f"Successfully opened {len(urls)} URLs in the browser."
    except Exception as e:
        return f"An error occurred: {e}"

# Tool Function: brave_search
def brave_search(query: str, offset: int = 0) -> str:
    """Retrieve search results using Brave Search API."""
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return "Error: API key not found. Please set the 'BRAVE_API_KEY' environment variable."

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    params = {
        "q": query,
        "offset": offset,
        "result_filter": "web"  # Use result_filter to include only web results
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()

            # Access the 'mixed' field and its 'main' list
            mixed_results = data.get("mixed", {}).get("main", [])
            extracted_results = []

            # Locate the actual search result data in the API response
            web_results = data.get("web", {}).get("results", [])

            for result in mixed_results:
                index = result.get("index")
                if index is not None and 0 <= index < len(web_results):
                    web_result = web_results[index]
                    title = web_result.get("title", "No title")
                    url = web_result.get("url", "No URL")
                    description = web_result.get("description", "No description")
                    extracted_results.append({
                        "title": title,
                        "url": url,
                        "description": description
                    })

            # Format the results as a numbered list
            result_list = []
            for idx, item in enumerate(extracted_results, start=1):
                result_list.append(f"{idx}. Title: {item['title']}")
                result_list.append(f"   URL: {item['url']}")
                result_list.append(f"   Description: {item['description']}\n")

            return "\n".join(result_list)
        else:
            return f"Error: Failed to retrieve data, status code: {response.status_code}\nDetails: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"


class SeleniumTextExtractor(BaseCase):
    
    def seleniumbase_bs4_extractor(self, url: str, output_file: str, selector: Optional[str] = None) -> str:
        """
        Loads a URL, uses SeleniumBase to clean and extract the full HTML content, then applies BS4 for parsing
        and extraction of main content (text and code). Optionally allows for a CSS selector to directly extract content.

        Parameters:
        - url (str): The URL to extract content from.
        - output_file (str): The path to the output text file where the extracted content will be saved.
        - selector (str, optional): The CSS selector for directly extracting content. If None, the BS4 process is used.

        Returns:
        - str: Success message indicating the text was saved to the specified file.
        """
        # Step 1: Open the specified URL using SeleniumBase
        self.open(url)

        # Step 2: Clean the page by removing unwanted elements
        self.remove_elements("style, script, header, footer, nav, aside, meta, link")

        # Step 3: If a custom selector is provided, try to extract using that selector
        if selector:
            try:
                page_html = self.get_html(selector)
            except Exception as e:
                page_html = f"Failed to extract content using the provided selector: {selector}. Error: {e}"
        else:
            # Step 4: Extract the full page HTML for BeautifulSoup parsing
            page_html = self.get_page_source()

        # Step 5: Use BeautifulSoup to parse the HTML content
        soup = BeautifulSoup(page_html, 'lxml')

        # Step 6: Use the BS4 method you implemented earlier to extract main content and code snippets
        main_container, largest_text_block = self.find_main_content_and_code(soup)
        code_snippets = self.extract_code_snippets(main_container)

        # Step 7: Save the extracted text and code to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Main Text:\n")
            f.write(largest_text_block + "\n\n")

            for i, code in enumerate(code_snippets, start=1):
                f.write(f"Code Block {i}:\n")
                f.write(code + "\n\n")

        return f"Successfully saved main content and code from '{url}' to '{output_file}'."

    # Define a Test Runner for SeleniumBase with BS4 Extraction
def run_selenium_bs4_extractor(urls: List[str], output_dir: str, selector: Optional[str] = None) -> str:
    """
    Runs the SeleniumBase loader function integrated with BS4 for text extraction on multiple URLs.

    Parameters:
    - urls (List[str]): A list of URLs to extract main content and code from.
    - output_dir (str): The directory where the output text files will be saved.
    - selector (str, optional): A CSS selector to directly extract content, if known.

    Returns:
    - str: Success message indicating the number of URLs processed and the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for url in urls:
        # Generate a unique output file name based on the URL
        parsed_url = urlparse(url)
        file_name = f"{parsed_url.netloc}_{parsed_url.path.replace('/', '_')}.txt"
        output_file = os.path.join(output_dir, file_name)

        temp_script = f"""
from seleniumbase import BaseCase
from bs4 import BeautifulSoup

class TempTextExtractor(BaseCase):
    def test_extract_main_text(self):
        self.open("{url}")
        self.remove_elements("style, script, header, footer, nav, aside, meta, link")
        page_html = self.get_page_source()

        # Use BeautifulSoup for text extraction
        soup = BeautifulSoup(page_html, 'lxml')

        # Call the BS4 methods for content extraction
        main_container, largest_text_block = self.find_main_content_and_code(soup)
        code_snippets = self.extract_code_snippets(main_container)

        # Write to the output file
        with open("{output_file}", "w", encoding="utf-8") as file:
            file.write("Main Text:\\n")
            file.write(largest_text_block + "\\n\\n")

            # Writing code blocks
            for i, code in enumerate(code_snippets, start=1):
                file.write(f"Code Block {{i}}:\\n")
                file.write(code + "\\n\\n")

        print(f"Successfully saved content from '{url}' to '{output_file}'.")

    def find_main_content_and_code(self, soup):
        largest_text_block = ""
        main_container = None
        largest_word_count = 0

        # Loop through all <div>, <article>, <section> tags
        potential_containers = soup.find_all(['div', 'article', 'section'])

        # Iterate through each potential container
        for container in potential_containers:
            text = container.get_text(separator=" ", strip=True)
            word_count = len(text.split())

            # Check if this container has more text than previous containers
            if word_count > largest_word_count:
                largest_word_count = word_count
                largest_text_block = text
                main_container = container

        return main_container, largest_text_block

    def extract_code_snippets(self, container):
        code_snippets = []
        if container:
            code_tags = container.find_all(['code', 'pre'])
            for code in code_tags:
                code_snippets.append(code.get_text(strip=True))
        return code_snippets

if __name__ == "__main__":
    from pytest import main
    main(["-v", __file__])
"""

        temp_script_path = "temp_selenium_test.py"

        with open(temp_script_path, "w") as file:
            file.write(temp_script)

        try:
            subprocess.run(["pytest", temp_script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to extract text from '{url}'. Error: {e}")
        finally:
            if os.path.exists(temp_script_path):
                os.remove(temp_script_path)

    return f"Successfully processed {len(urls)} URLs. Extracted content saved in '{output_dir}'."
 

# Define the Pydantic model for Code Blog Metadata
class CodeBlogMetadata(BaseModel):
    """Structured output model for blog posts containing code snippets and explanations."""
    
    code_snippets: List[str] = Field(..., description="List of code snippets retrieved from the blog post.")
    explanations: List[str] = Field(..., description="Text explanations related to the code snippets.")
    entities: List[str] = Field(..., description="List of entities extracted from the blog post.")

# Global variables for query engine and index
query_engine = None
index = None

def query_index(query_text: str) -> str:
    """
    Loads documents from the './data' directory, creates an index, saves it to './storage', and queries the index.
    If './storage' already exists, it loads the index from storage.
    In both cases, it queries the index with the provided query_text.

    Parameters:
    - query_text (str): The text query to search in the index.

    Returns:
    - str: The formatted response with marvin_metadata.
    """
    global index, query_engine

    directory_path = "./data"
    storage_path = "./storage"

    # Check if the storage directory already exists
    if not os.path.exists(storage_path):
        # Load the documents and create the index if storage doesn't exist
        documents = SimpleDirectoryReader(directory_path).load_data()
        if not documents:
            return f"Error: No documents found in directory '{directory_path}'."
        
        # Define the metadata extraction and text splitting tools
        text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)
        qa_extractor = QuestionsAnsweredExtractor(questions=3)

        # Create MarvinMetadataExtractor with custom CodeBlogMetadata
        marvin_metadata_extractor = MarvinMetadataExtractor(
            marvin_model=CodeBlogMetadata  # Extract custom entities: code snippets, explanations, entities
        )

        # Create a new index with metadata extraction transformations
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[
                text_splitter,
                qa_extractor,
                marvin_metadata_extractor
            ]
        )

        # Save the index to disk
        index.set_index_id("vector_index")
        index.storage_context.persist(storage_path)
    else:
        # Load the index from storage if it already exists
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context, index_id="vector_index")

    # Create the query engine
    query_engine = index.as_query_engine(similarity_top_k=3)

    # Query the loaded index
    try:
        # Retrieve nodes from the query engine
        response = query_engine.query(query_text)
        
        # Retrieve the source nodes from the response
        retrieved_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []

        if not retrieved_nodes:
            return "No relevant information found in the index."

        # Format the response with marvin_metadata
        formatted_response = f"Query: {query_text}\nResponse:\n"
        for i, node_with_score in enumerate(retrieved_nodes):
            node_content = node_with_score.node.get_content()
            metadata = node_with_score.node.extra_info.get("marvin_metadata", {})

            # Extract marvin_metadata fields
            code_snippets = metadata.get('code_snippets', [])
            explanations = metadata.get('explanations', [])
            entities = metadata.get('entities', [])

            # Format the output
            formatted_response += f"{i+1}. Content: {node_content}\n"
            formatted_response += f"Score: {node_with_score.score}\n"
            formatted_response += f"Code Snippets: {', '.join(code_snippets) if code_snippets else 'None'}\n"
            formatted_response += f"Entities: {', '.join(entities) if entities else 'None'}\n\n"

        return formatted_response
    except Exception as e:
        return f"An error occurred: {e}"