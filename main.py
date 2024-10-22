import os
from autogen import ConversableAgent, GroupChat, GroupChatManager, Cache
from tools import brave_search, open_browser, run_selenium_bs4_extractor, query_index  

# Initialize the agent that searches on the web using the Brave Search API
brave_search_agent = ConversableAgent(
    name="brave_search_agent",
    system_message=(
        "Your name is BraveSearchAgent. You are responsible for searching the web using the Brave Search API. "
        "You can retrieve search results based on user queries and provide back the top 20 links from the results. "
        "To avoid rate limits, please only call the brave_search tool once per user query."
    ),
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY"), "max_tokens": 1024}]},
    code_execution_config=False,
    function_map=None,
    human_input_mode="NEVER",
)


# Initialize the user proxy agent
user_proxy = ConversableAgent(
    name="UserProxy",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="ALWAYS",
)

# Register the brave_search tool with the agent
brave_search_agent.register_for_llm(name="brave_search", description="Searches the web using the Brave Search API.")(brave_search)

# Register the brave_search tool for execution with the user_proxy agent
user_proxy.register_for_execution(name="brave_search")(brave_search)

# Initialize a new agent for opening browsers
browser_agent = ConversableAgent(
    name="BrowserAgent",
    system_message=(
        "Your name is BrowserAgent. You take input from the user to open URLs in a web browser. "
        "You specifically open URLs in Google Chrome, either in new windows or tabs."
    ),
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}]},
    code_execution_config=False,
    function_map=None,
    human_input_mode="NEVER",
)

# Register the open_browser tool with the browser_agent
browser_agent.register_for_llm(name="open_browser", description="Opens a URL in a web browser.")(open_browser)

# Register the open_browser tool for execution with the user_proxy agent
user_proxy.register_for_execution(name="open_browser")(open_browser)

# Initialize a new agent for running Selenium and BS4 extraction
selenium_extractor_agent = ConversableAgent(
    name="SeleniumExtractorAgent",
    system_message=(
        "Your name is SeleniumExtractorAgent. You are responsible for extracting main content and code from web pages "
        "using Selenium and BeautifulSoup."
    ),
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}]},
    code_execution_config=False,
    function_map=None,
    human_input_mode="NEVER",
)

# Register the run_selenium_bs4_extractor tool with the selenium_extractor_agent
selenium_extractor_agent.register_for_llm(name="run_selenium_bs4_extractor", description="Extracts main content and code from a web page using Selenium and BeautifulSoup.")(run_selenium_bs4_extractor)

# Register the run_selenium_bs4_extractor tool for execution with the user_proxy agent
user_proxy.register_for_execution(name="run_selenium_bs4_extractor")(run_selenium_bs4_extractor)


# Initialize a new agent for querying an index
query_index_agent = ConversableAgent(
    name="QueryIndexAgent",
    system_message=(
        "Your name is QueryIndexAgent. You are responsible for querying an index created from documents in the ./data folder. "
        "You can retrieve relevant information based on user queries using the query_index tool."
    ),
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}]},
    code_execution_config=False,
    function_map=None,
    human_input_mode="NEVER",
)

# Register the query_index tool with the QueryIndexAgent
query_index_agent.register_for_llm(name="query_index", description="Queries an index created from documents in the ./data folder.")(query_index)

# Register the query_index tool for execution with the user_proxy agent
user_proxy.register_for_execution(name="query_index")(query_index)

# New agent for structuring the response
response_structurer_agent = ConversableAgent(
    name="ResponseStructurerAgent",
    system_message=(
        "Your name is ResponseStructurerAgent. You take the output from the QueryIndexAgent and structure it into a "
        "readable format, focusing on code snippets and their explanations. Ensure that the code snippets are complete "
        "and properly formatted. Your response should be clear, concise, and easy to understand."
    ),
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ.get("OPENAI_API_KEY")}]},
    code_execution_config=False,
    function_map=None,
    human_input_mode="NEVER",
)

# Create a GroupChat object with the list of agents
group_chat = GroupChat(
    agents=[brave_search_agent, browser_agent, user_proxy, selenium_extractor_agent, query_index_agent, response_structurer_agent],
    messages=[],
    max_round=14,
)

# Create a GroupChatManager object and provide the GroupChat object as input
group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]}
)


# Example usage
def agent_interaction():
    """
    
    print("Tool schema for brave_search_agent :")
    print(brave_search_agent.llm_config.get("tools", "No tools registered"))

    print("Tool schema for browser_agent :")
    print(browser_agent.llm_config.get("tools", "No tools registered"))

    
    print("Tool schema for selenium_extractor_agent :")
    print(selenium_extractor_agent.llm_config.get("tools", "No tools registered"))

    """
    
    result = user_proxy.initiate_chat(
        group_chat_manager,
        message="query the index on how to generate function arguments",
        summary_method="reflection_with_llm"
    )

if __name__ == "__main__":
    agent_interaction()