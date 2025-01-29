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
        
    def fix_text(self, text, verbose=False):
      
      query = f"Dado el siguiente texto que fue extraido utilizando tesseract y tiene errores por la deficiente calidad del texto: {text}. \n el texto correcto y sin formato debe ser: "
      return self._ask_model(query, verbose)
        
    def _ask_model(self, query, verbose):
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        
        response = completion.choices[0].message.content.strip().split("\n\n")[0]
        
        if verbose:
            print(response)
        
        return response
        
        
if __name__ == "__main__":
    
    llm = OpenAIModel()
    llm.fix_text("El pero do Donel tene hambre", verbose=True)