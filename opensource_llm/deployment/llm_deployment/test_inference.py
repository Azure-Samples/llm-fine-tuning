"""Simple script to summarize a markdown document and extract key points using a hosted LLM.

Reads the file `sample_doc.md` in the same directory, constructs a summarization
prompt, and requests a response limited to <= 500 tokens.
"""

import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv()

DOC_PATH = os.path.join(os.path.dirname(__file__), "sample_doc.md")

api_key = os.getenv("ENDPOINT_KEY")
api_base = os.getenv("ENDPOINT_URL")
# url = f"{api_base.rstrip('/')}/v1/chat/completions"


if not api_key or not api_base:
    raise RuntimeError("ENDPOINT_KEY and ENDPOINT_URL environment variables must be set.")

extra_headers = {
    "Authorization": f"Bearer {api_key}",
    "azureml-model-deployment": "vllm-llama2",
}

client = OpenAI(
    api_key=api_key,  # actual key passed via header for this endpoint style
    base_url=api_base,
)

with open(DOC_PATH, "r", encoding="utf-8") as f:
    document_text = f.read()

system_instructions = (
    "You are an expert legal and technical summarization assistant. "
    "Given a contractual / technical hosting agreement in markdown, produce: \n"
    "1. A concise executive summary (3-5 sentences).\n"
    "2. Bullet list of 8-12 most critical key points (obligations, SLAs, data, IP).\n"
    "3. Key risks or negotiation watch-outs (max 5).\n"
    "4. Structured JSON object at the end with fields: { 'uptime_sla', 'latency_sla', 'data_breach_notice_hours', 'governing_law_placeholder', 'termination_notice_days' }.\n"
    "Keep the whole response under 500 tokens. Be precise and avoid repetition."
)

# Because the model is text-only here, we provide the document inline. If the document is large,
# we could chunk or truncate; for now we send full text (optionally you may slice if size issues occur).
user_prompt = f"SOURCE DOCUMENT START\n{document_text}\nSOURCE DOCUMENT END"

messages = [
    {"role": "system", "content": system_instructions},
    {"role": "user", "content": user_prompt},
]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-1B-Instruct",  # switched to non-VL instruct model for text summary
    messages=messages,
    extra_headers=extra_headers,
    max_tokens=500,
    temperature=0.3,
)

print(response)

try:
    # If standard OpenAI-style structure
    print("\n--- Summary Output ---\n")
    print(response.choices[0].message.content)
except Exception:
    pass