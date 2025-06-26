echo "# ops-agent-" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:jurgen-paul/ops-agent-.git
git push -u origin main 

from huggingface_hub import InferenceClient
from e2b_code_interpreter import Sandbox
import re

# Not all models are capable of direct tools use - we need to extract the code block manually and prompting the LLM to generate the code.
def match_code_block(llm_response):
  pattern = re.compile(r'```python\n(.*?)\n```', re.DOTALL) # Match everything in between ```python and ```
  match = pattern.search(llm_response)
  if match:
    code = match.group(1)
    print(code)
    return code
  return ""


system_prompt = """You are a helpful coding assistant that can execute python code in a Jupyter notebook. You are given tasks to complete and you run Python code to solve them.
Generally, you follow these rules:
- ALWAYS FORMAT YOUR RESPONSE IN MARKDOWN
- ALWAYS RESPOND ONLY WITH CODE IN CODE BLOCK LIKE THIS:
\`\`\`python
{code}
\`\`\`
"""
prompt = "Calculate how many r's are in the word 'strawberry.'"

# Initialize the client
client = InferenceClient(
    provider="hf-inference",
    api_key="HF_INFERENCE_API_KEY"
)

completion = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B", # Or use any other model from Hugging Face
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
)

content = completion.choices[0].message.content
code = match_code_block(content)

with Sandbox() as sandbox:
    execution = sandbox.run_code(code)
    print(execution)
