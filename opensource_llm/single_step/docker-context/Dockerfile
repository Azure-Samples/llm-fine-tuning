# Use the base image from Azure ML or another suitable base image  
FROM mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:104  

# Install any system-level dependencies if needed  
RUN apt-get update && apt-get install -y \  
    build-essential \  
    && rm -rf /var/lib/apt/lists/*  

# Set the working directory  
WORKDIR /app  

# Install Python and pip dependencies  
RUN  pip install --upgrade pip  

# Copy requirements.txt and install Python packages  
COPY requirements.txt .  

# Install the main Python packages  
RUN pip install -r requirements.txt  

# Install flash-attn with no build isolation  
RUN pip install flash-attn --no-build-isolation  
CMD ["bash"]
