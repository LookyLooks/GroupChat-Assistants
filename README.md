# 🤖 Multi-Agent Conversational AI with Tool Integration

## Project Title
**Multi-Agent Conversational AI with Tool Integration**

## Project Description
This project demonstrates the integration of multiple conversational AI agents with various tools to perform complex tasks such as web searching, content extraction, and document indexing. The system leverages the `autogen` library to facilitate multi-agent conversations, enabling sophisticated workflows with minimal effort. It supports human feedback to guide the agents, ensuring they meet user goals effectively.

### Key Features
- **Multi-Agent Conversations**: Utilize `autogen` to build next-gen LLM applications based on multi-agent conversations, supporting diverse conversation patterns for complex workflows.
- **GroupChat Functionality**: Implement a more deterministic agent workflow using GroupChat.
- **Human-in-the-Loop**: Allow human feedback to steer agents, specify goals, and refine outputs throughout the process.
- **Tool Integration**: Agents can call various tools to perform actions like web searching, text scraping, file reading, and RAG (Retrieval-Augmented Generation).
- **Embedding and Similarity Search**: Create embeddings, store them, and perform similarity searches on scraped pages.

## Table of Contents
1. [Installation](#installation)
2. [Environment Setup](#environment-setup)
3. [How It Works](#how-it-works)
4. [How to Use the Project](#how-to-use-the-project)
5. [Agent Descriptions](#agent-descriptions)
6. [Example Workflow](#example-workflow)
7. [Documentation Links](#documentation-links)

## Installation
To set up the project, ensure you have Python 3.8 or higher installed and run the following commands:

```bash
pip install seleniumbase
pip install beautifulsoup4
pip install llama-index
pip install llama-index-extractors-marvin
pip install autogen-agentchat~=0.2
```

## Environment Setup
The code uses the following environment variables:
```bash
export BRAVE_API_KEY=your_brave_api_key
export OPENAI_API_KEY=your_openai_api_key
```

## How It Works
The code in `main.py` initializes several agents, each with specific responsibilities, and integrates them into a `GroupChat` for coordinated task execution. The conversation is initiated with the message in the `initiate_chat` method, and the code is run with `python main.py`. The agents can perform actions such as:

- 🌐 **Web Searching**: Using the Brave Search API to find relevant information.
- 🖥️ **Browser Automation**: Opening URLs in a web browser.
- 🕵️ **Content Extraction**: Scraping text and code from web pages using Selenium and BeautifulSoup.
- 📚 **Document Indexing**: Creating and querying a local index from documents.
- 🧠 **RAG Processing**: Creating embeddings, storing them, and performing similarity searches on scraped content.

## How to Use the Project

## Agent Descriptions
- **BraveSearchAgent**: Searches the web using the Brave Search API and returns the top 20 links.
- **BrowserAgent**: Opens URLs in Google Chrome, either in new windows or tabs.
- **SeleniumExtractorAgent**: Extracts main content and code from web pages using Selenium and BeautifulSoup.
- **QueryIndexAgent**: Queries an index created from documents in the `./data` folder to retrieve relevant information.
- **ResponseStructurerAgent**: Processes the output from the QueryIndexAgent and structures it into a readable format, focusing on complete code snippets and their explanations.

## Example Workflow
1. **Search for Information**: The `BraveSearchAgent` uses the Brave Search API to find relevant web content.
2. **Open Links**: The `BrowserAgent` opens specific links in a browser window.
3. **Extract Content**: The `SeleniumExtractorAgent` scrapes content from web pages.
4. **Index and Query**: The `QueryIndexAgent` creates an index from the extracted content, generates embeddings, and allows querying for specific information using similarity search.
5. **Structure Response**: The `ResponseStructurerAgent` takes the output from the QueryIndexAgent and structures it into a readable format, focusing on complete code snippets and their explanations.

## Human-in-the-Loop
The system supports continuous human feedback, allowing users to guide the agents, specify goals, and refine outputs throughout the process. This ensures that the agents' actions align with user expectations and objectives, and allows for real-time adjustments to the workflow.

## Documentation Links
- **AutoGen Documentation**: [AutoGen Getting Started](https://microsoft.github.io/autogen/0.2/docs/Getting-Started)
- **LlamaIndex Documentation**: [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/)
- **SeleniumBase Documentation**: [SeleniumBase Documentation](https://seleniumbase.io/)

## Conclusion
This project showcases the power of integrating multiple conversational AI agents with tool-based actions, enabling complex workflows through a single, coordinated interface. By leveraging `autogen` and implementing a GroupChat structure, the system provides a flexible and scalable solution for automating sophisticated research and information-processing tasks with built-in human oversight.

---

Feel free to explore the code in `main.py` to understand how each agent is initialized and how they interact within the `GroupChat`. Happy coding! 🚀
