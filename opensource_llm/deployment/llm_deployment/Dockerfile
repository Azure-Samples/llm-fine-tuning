FROM vllm/vllm-openai:latest  
  
# Set environment variables  
ENV MODEL_NAME meta-llama/Llama-3.2-1B-Instruct 
  
# Install required Python libraries  
RUN pip install transformers \  
    && pip install accelerate \  
    && pip install 'vllm>0.10.0'  
  
# Execute vLLM server entrypoint with the specified model and arguments  
ENTRYPOINT python3 -m vllm.entrypoints.openai.api_server --model $MODEL_NAME $VLLM_ARGS  
