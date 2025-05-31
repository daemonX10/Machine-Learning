from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(
    model_name='claude-3-sonnet-20240229',
    timeout=10,
    temperature=0.5,
    stop=['\n\n']
)

result = model.invoke("how to revise machine learning ?")

print(result.content)