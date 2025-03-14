from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
import json
from tools import search_tool, wiki_tool, save_tool


load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccesssary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

tools=[search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

query = input("What can I help you research? ")
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
raw_response = agent_executor.invoke({"query": query})

# Extract the "output" field from the raw response
raw_output = raw_response.get("output")

# Check if the output is a string
if isinstance(raw_output, str):
    # Strip the markdown code block part (` ```json` and closing backticks)
    cleaned_output = raw_output.strip("```json\n").strip("```")

    try:
        # Parse the cleaned-up JSON string into a dictionary
        parsed_output = json.loads(cleaned_output)

        # Access properties of the dictionary
        print("Topic:", parsed_output.get("topic"))
        print("Summary:", parsed_output.get("summary"))
        print("Sources:", parsed_output.get("sources"))
        print("Tools Used:", parsed_output.get("tools_used"))
        
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
else:
    print("The output is not a string, it's already a dictionary.")


