from llama_cpp import Llama
import sys

def get_text_from_url(url = 'https://docs.cloudera.com/cdp-private-cloud-base/7.1.9/runtime-release-notes/topics/rt-pvc-fixed-issues-hadoop.html'):
    import requests
    from bs4 import BeautifulSoup
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text()
        return text_content
    else:
        print(f"Failed to retrieve the web page. Status code: {response.status_code}")
        return ""

def get_text_from_file(file = '1-chat.py'):
    with open(file, 'r') as f:
        text = f.read()
        return text
      
      
content = get_text_from_url()
prompt = content + """
-------
Based on the information above, provide a list of jiras mentioned in text.
"""
print(prompt)
LLM = Llama(model_path="models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", 
            n_ctx=512, n_gpu_layers=32)
output = LLM(prompt, max_tokens=128, echo=False)
print("------------ LLM Response ------------")
print(output["choices"][0]["text"].strip())



content = get_text_from_file()
prompt = content + """
-------
Provide the analysis of Python code given above.
"""
print(prompt)
LLM = Llama(model_path="models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", 
            n_ctx=512, n_gpu_layers=32)
output = LLM(prompt, max_tokens=2048, echo=False)
print("------------ LLM Response ------------")
print(output["choices"][0]["text"].strip())
