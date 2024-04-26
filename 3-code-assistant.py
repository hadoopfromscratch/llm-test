from llama_cpp import Llama
import re
  
def parse_and_save(answer):
    try:
        pattern = r'```python(.*?)```'
        code = re.findall(pattern, answer, re.DOTALL)[0]
        print("--------------PYTHON-----------------")
        print(code)
        print("--------------PYTHON-----------------")
    except:
        print("ERROR! No script found")
        return
    with open('generated_python_script.py', 'w') as file:
        file.write(code)
    
prompt = """Write python script that computes factorial of input taken from the user."""
prompt = """Write python script that says hello 100 times."""

LLM = Llama(model_path="models/Phi-3-mini-4k-instruct-q4.gguf", 
            n_ctx=512, n_gpu_layers=32)
output = LLM(prompt, max_tokens=512, echo=False)
answer = output["choices"][0]["text"].strip()
print(answer)
parse_and_save(answer)