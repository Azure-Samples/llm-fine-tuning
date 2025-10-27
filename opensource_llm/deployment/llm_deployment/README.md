## Deploy vLLM for Qwen/Qwen2.5-VL-32B-Instruct on Azure Machine Learning (Managed Online Endpoints)

This guide adapts Clemens Siebler’s post “Deploying vLLM models on Azure Machine Learning with Managed Online Endpoints” to the Qwen/Qwen2.5-VL-32B-Instruct VLM using the files in this folder.

Reference: https://clemenssiebler.com/posts/vllm-on-azure-machine-learning-managed-online-endpoints-deployment/

What you’ll deploy
- vLLM OpenAI-compatible server hosting Qwen/Qwen2.5-VL-32B-Instruct
- Managed Online Endpoint using instance type Standard_NC24ads_A100_v4 (1x A100 80GB)

Prerequisites
- Azure CLI with ML extension (`az extension add -n ml`)
- Access to an Azure ML Workspace and Resource Group
- Optional: a Hugging Face access token if the model requires gated access (set via HUGGING_FACE_HUB_TOKEN)

Folder contents (tailored for Qwen2.5-VL-32B-Instruct)
- `Dockerfile`: vLLM base image; sets `MODEL_NAME=Qwen/Qwen2.5-VL-32B-Instruct` and installs qwen-vl-utils
- `environment.yml`: Azure ML Environment spec building the Dockerfile
- `endpoint.yml`: Managed Online Endpoint (name: `vllm-qwen2dot5`, auth: key)
- `vllm_deployment.yml`: Managed Online Deployment for Qwen2.5-VL-32B-Instruct

Notes on configuration
- Compute SKU: `Standard_NC24ads_A100_v4` is required for the 32B VL model
- vLLM args: `--max-model-len 15500 --enforce-eager --dtype bfloat16` (tune if needed)
- Concurrency: starts with `max_concurrent_requests_per_instance: 1` for stability on a single A100
- Optional HF token: if needed (not neccessary for QWEN model), add `HUGGING_FACE_HUB_TOKEN` under `environment_variables` in `vllm_deployment.yml`

---

## 1) Login and set defaults

```bash
az login
az account set --subscription "<SUBSCRIPTION_ID>"
az configure --defaults workspace="<AML_WORKSPACE_NAME>" group="<RESOURCE_GROUP>"
```

## 2) Build the custom vLLM environment

This builds the image defined by the local `Dockerfile` via Azure ML.

```bash
az ml environment create --file environment.yml
```

After creation, note the Docker image address of this environment (Azure ML Studio → Environments → `llm_deployment_env`).
Alternatively via CLI (replace VERSION with the created version):

```bash
az ml environment show --name llm_deployment_env --version <VERSION> --query image -o tsv
```

If needed, replace `environment.image` in `vllm_deployment.yml` with the retrieved image address.

## 3) Create the Managed Online Endpoint

```bash
az ml online-endpoint create --name vllm-qwen2dot5 -f endpoint.yml
```

## 4) Deploy the model to the endpoint

`vllm_deployment.yml` is preconfigured for Qwen/Qwen2.5-VL-32B-Instruct and `Standard_NC24ads_A100_v4`.

```bash
az ml online-deployment create -f vllm_deployment.yml --all-traffic
```

To adjust authentication for gated models (not neccessery for QWEN model), add to `vllm_deployment.yml` under `environment_variables`:

```yaml
  HUGGING_FACE_HUB_TOKEN: "<YOUR_HF_TOKEN>"
```

## 5) Get scoring URI and keys

```bash
az ml online-endpoint show -n vllm-qwen2dot5 -o jsonc
az ml online-endpoint get-credentials -n vllm-qwen2dot5 -o jsonc
```

You’ll use the `scoringUri` and one of the `primaryKey`/`secondaryKey` values below.

## 6) Test the endpoint

The vLLM server exposes an OpenAI-compatible API. For VLM, prefer `/v1/chat/completions` with image content.

Python (text-only chat)
```python
import requests

endpoint = "<SCORING_URI>"  # e.g., https://vllm-qwen2dot5.<region>.inference.ml.azure.com
api_key = "<KEY>"

url = f"{endpoint}/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
payload = {
	"model": "Qwen/Qwen2.5-VL-32B-Instruct",
	"messages": [
		{"role": "user", "content": [{"type": "text", "text": "Summarize San Francisco in one sentence."}]}
	],
	"max_tokens": 200,
	"temperature": 0.7
}

resp = requests.post(url, headers=headers, json=payload, timeout=180)
print(resp.status_code)
print(resp.json())
```

Python (vision: remote image URL)
```python
import requests

endpoint = "<SCORING_URI>"
api_key = "<KEY>"

url = f"{endpoint}/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
payload = {
	"model": "Qwen/Qwen2.5-VL-32B-Instruct",
	"messages": [
		{
			"role": "user",
			"content": [
				{"type": "text", "text": "Describe the road signs in this image."},
				{"type": "image_url", "image_url": {"url": "https://example.com/road.jpg"}}
			]
		}
	],
	"max_tokens": 256
}

resp = requests.post(url, headers=headers, json=payload, timeout=180)
print(resp.status_code)
print(resp.json())
```

Python (vision: local image → base64)
```python
import base64, requests

endpoint = "<SCORING_URI>"
api_key = "<KEY>"
image_path = "../data/Run 7_Camera 4 360_84_2.png"  # adjust path as needed

with open(image_path, "rb") as f:
	b64 = base64.b64encode(f.read()).decode("utf-8")

url = f"{endpoint}/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
payload = {
	"model": "Qwen/Qwen2.5-VL-32B-Instruct",
	"messages": [
		{
			"role": "user",
			"content": [
				{"type": "text", "text": "List and classify any visible road signs."},
				{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
			]
		}
	],
	"max_tokens": 256
}

resp = requests.post(url, headers=headers, json=payload, timeout=180)
print(resp.status_code)
print(resp.json())
```

## Throughput and reliability tips
- Instance type: `Standard_NC24ads_A100_v4` is required; consider starting with `max_concurrent_requests_per_instance: 1` and increase cautiously.
- Tune `request_timeout_ms` based on prompt size and image processing time; large images/long contexts need more time.
- If you see HTTP 429/timeouts, reduce concurrency or implement client-side retry with exponential backoff.

## Autoscaling (optional)
Managed Online Endpoints can autoscale via Azure Monitor rules (GPU/CPU utilization or schedules). See: https://learn.microsoft.com/azure/machine-learning/how-to-autoscale-managed-online-endpoints

## Cleanup
```bash
az ml online-endpoint delete --name vllm-qwen2dot5 --yes --no-wait
```

—
Based on Clemens Siebler’s blog post; adapted to Qwen/Qwen2.5-VL-32B-Instruct and the configs in this folder.
