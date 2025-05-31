from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os 

# Set the cache directory
os.environ['HF_HOME'] = "D:/huggingface_cache"
os.environ['TRANSFORMERS_CACHE'] = "D:/huggingface_cache"

# Use the model ID instead of local path - it will use your cached version
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Use model ID, not local path
    task="text-generation",
    model_kwargs={
        "temperature": 0.1,
        "max_new_tokens": 256,
        "do_sample": True,
        "repetition_penalty": 1.1,
    },
    pipeline_kwargs={
        "return_full_text": False,
        "pad_token_id": 2,  # Move pad_token_id here
    }
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Why is machine learning important?")

print(result.content)