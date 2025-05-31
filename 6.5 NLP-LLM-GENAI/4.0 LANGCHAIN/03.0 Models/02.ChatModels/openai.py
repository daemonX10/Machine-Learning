from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

model = ChatOpenAI(
    model='gpt-4o',
    temperature=0.5 ,
    max_completion_tokens=10
)

result = model.invoke("how to revise machine learning ?")

print(result.content)