from llama_cpp import Llama, LlamaGrammar
import sys

def sentiment(msg, LLM):
    prompt = f"[INST]Answer 'positive', 'neutral', or 'negative' only depending on the sentiment of the message.[/INST]ok[INST]This bike is a piece of junk.[/INST]negative[INST]{msg}[/INST]"
    grammar_text = 'root    ::= ("positive" | "neutral" | "negative")'
    grammar = LlamaGrammar.from_string(grammar_text)
    output = LLM(prompt, grammar=grammar, max_tokens=16, echo=False)
    #output = LLM(prompt, max_tokens=16, echo=True)
    return output["choices"][0]["text"].strip()
  
tweets = ["Love this film",
          "No way I'm buying this car again",
          "I'm going to the office today"]

LLM = Llama(model_path="models/mixtral-8x7b-v0.1.Q4_K_M.gguf", 
            n_ctx=128, n_gpu_layers=28)


print("------------ LLM Response ------------")
for tweet in tweets:
    print(f"{tweet} ------> {sentiment(tweet, LLM)}")