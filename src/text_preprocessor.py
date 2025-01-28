import os
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("API_KEY")


class OpenAIModel:

    def __init__(self, model="gemma2_instruct_2b_es"):

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY,
        )
        
        self.model = "google/gemma-2-9b-it:free"
        
    def ask_model(self, query, verbose=False):
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        
        response = completion.choices[0].message.content
        
        if verbose:
            print(response)
        
        return response
        
        
if __name__ == "__main__":
    
    llm = OpenAIModel()
    llm.ask_model("Completa la siguiente frase: Solo el amor convierte en milagro...", verbose=True)
    llm.ask_model("De quien es la frase 'Solo el amor convierte ...'?", verbose=True)
    
    
'''
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="<OPENROUTER_API_KEY>",
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  model="google/gemma-2-9b-it:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)

import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer <OPENROUTER_API_KEY>",
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "google/gemma-2-9b-it:free", # Optional
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
    
  })
)
'''