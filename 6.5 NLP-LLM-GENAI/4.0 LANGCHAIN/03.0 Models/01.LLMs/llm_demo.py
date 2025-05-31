from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os


load_dotenv()

llm = GoogleGenerativeAI(
    model ='gemini-1.5-flash'
)

result = llm.invoke('why should oops is learn in java not c++ ?')

print(result)
