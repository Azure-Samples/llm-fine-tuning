from __future__ import annotations  
import argparse  
import asyncio  
import json  
import os  
import random  
import string  
import statistics  
import time  
import math  
from dataclasses import dataclass, field  
from typing import List, Optional, Dict, Any  
  
import dotenv  
import httpx  
  
dotenv.load_dotenv()  
  
@dataclass  
class RequestResult:  
    latency: float  
    success: bool  
    error: Optional[str] = None  
    output_tokens: Optional[int] = None  
    input_tokens: Optional[int] = None  
    start_time: Optional[float] = None  
    end_time: Optional[float] = None  
  
@dataclass  
class Metrics:  
    results: List[RequestResult] = field(default_factory=list)  
    configured_concurrency: Optional[int] = None  
    _start_time: Optional[float] = None  
    _end_time: Optional[float] = None  
  
    def start_timer(self):  
        self._start_time = time.perf_counter()  
  
    def end_timer(self):  
        self._end_time = time.perf_counter()  
  
    def add(self, rr: RequestResult) -> None:  
        self.results.append(rr)  
  
    def _percentile(self, data: List[float], p: float) -> Optional[float]:  
        """Precise percentile calculation using linear interpolation."""  
        if not data:  
            return None  
        data_sorted = sorted(data)  
        k = (len(data_sorted) - 1) * (p / 100)  
        f = math.floor(k)  
        c = math.ceil(k)  
        if f == c:  
            return data_sorted[int(k)]  
        return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)  
  
    def summary(self) -> Dict[str, Any]:  
        if not self.results:  
            return {  
                "configured_concurrency": self.configured_concurrency,  
                "duration_s": None  
            }  
  
        total_wall_time = None  
        if self._start_time and self._end_time:  
            total_wall_time = self._end_time - self._start_time  
  
        successes = [r for r in self.results if r.success]  
        failures = [r for r in self.results if not r.success]  
        latencies = [r.latency for r in successes]  
  
        output_tokens = [r.output_tokens for r in successes if r.output_tokens is not None]  
        input_tokens = [r.input_tokens for r in successes if r.input_tokens is not None]  
  
        return {  
            "total_requests": len(self.results),  
            "successful": len(successes),  
            "failed": len(failures),  
            "success_rate": (len(successes) / len(self.results)) * 100,  
            "rps": (len(self.results) / total_wall_time) if total_wall_time else None,  
            "successful_rps": (len(successes) / total_wall_time) if total_wall_time else None,  
            "latency_avg_s": statistics.fmean(latencies) if latencies else None,  
            "latency_min_s": min(latencies) if latencies else None,  
            "latency_max_s": max(latencies) if latencies else None,  
            "latency_p50_s": self._percentile(latencies, 50),  
            "latency_p90_s": self._percentile(latencies, 90),  
            "latency_p95_s": self._percentile(latencies, 95),  
            "latency_p99_s": self._percentile(latencies, 99),  
            "avg_output_tokens": statistics.fmean(output_tokens) if output_tokens else None,  
            "avg_input_tokens": statistics.fmean(input_tokens) if input_tokens else None,  
            "throughput_output_tokens_per_s": (sum(output_tokens) / total_wall_time) if output_tokens and total_wall_time else None,  
            "throughput_input_tokens_per_s": (sum(input_tokens) / total_wall_time) if input_tokens and total_wall_time else None,  
            "configured_concurrency": self.configured_concurrency,  
            "duration_s": total_wall_time  
        }  
  
def append_random_suffix_to_prompt(payload: Dict[str, Any]) -> Dict[str, Any]:  
    """Safely append 10 random alphanumeric chars to the first user prompt."""  
    rand_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=10))  
    payload_copy = json.loads(json.dumps(payload))  # deep copy  
  
    try:  
        first_msg = payload_copy["messages"][0]  
        content = first_msg["content"]  
        if isinstance(content, list):  
            if content and isinstance(content[0], dict) and "text" in content[0]:  
                content[0]["text"] += f" [{rand_suffix}]"  
        elif isinstance(content, str):  
            first_msg["content"] += f" [{rand_suffix}]"  
    except Exception as e:  
        print(f"[WARN] Failed to append random suffix: {e}")  
  
    return payload_copy  
  
async def make_request(client: httpx.AsyncClient, url: str, headers: Dict[str, str],  
                       payload: Dict[str, Any], timeout: float, retries: int = 2) -> RequestResult:  
    start_time = time.perf_counter()  
    last_error: Optional[str] = None  
  
    for attempt in range(retries + 1):  
        try:  
            resp = await client.post(url, headers=headers, json=payload, timeout=timeout)  
            if resp.status_code >= 500:  
                last_error = f"Server {resp.status_code}"  
                await asyncio.sleep(0.2 * (attempt + 1))  
                continue  
            if resp.status_code != 200:  
                last_error = f"HTTP {resp.status_code}: {resp.text[:120]}"  
                break  
            data = resp.json()  
            usage = data.get("usage", {})  
            out_tokens = usage.get("completion_tokens")  
            in_tokens = usage.get("prompt_tokens")  
            end_time = time.perf_counter()  
            return RequestResult(  
                latency=end_time - start_time,  
                success=True,  
                output_tokens=out_tokens,  
                input_tokens=in_tokens,  
                start_time=start_time,  
                end_time=end_time  
            )  
        except Exception as e:  
            last_error = str(e)  
            await asyncio.sleep(0.2 * (attempt + 1))  
            continue  
  
    end_time = time.perf_counter()  
    return RequestResult(  
        latency=end_time - start_time,  
        success=False,  
        error=last_error,  
        start_time=start_time,  
        end_time=end_time  
    )  
  
async def worker(stop_event: asyncio.Event, metrics: Metrics, client: httpx.AsyncClient,  
                 url: str, headers: Dict[str, str], base_payload: Dict[str, Any], timeout: float):  
    while not stop_event.is_set():  
        payload = append_random_suffix_to_prompt(base_payload)  
        rr = await make_request(client, url, headers, payload, timeout)  
        
        metrics.add(rr)  
  
async def run_benchmark(args) -> Dict[str, Any]:  
    endpoint = os.getenv("ENDPOINT_URL", "http://localhost:8000")  
    api_key = os.getenv("ENDPOINT_KEY")  
    url = f"{endpoint.rstrip('/')}/v1/chat/completions"  
    headers = {  
        "Content-Type": "application/json",  
        "Authorization": f"Bearer {api_key}",  
        "azureml-model-deployment": "vllm-llama3"  
    }  
  
    prompt_text = "Summarize this contract to 200 words \n"  
    if args.prompt_file:  
        with open(args.prompt_file, "r", encoding="utf-8") as pf:  
            prompt_text += pf.read().strip()  
  
    payload = {  
        "model": args.model,  
        "messages": [  
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]},  
        ],  
        "max_tokens": args.max_tokens,  
        "temperature": args.temperature,  
    }  
  
    timeout = args.request_timeout  
  
    # Warmup (not recorded in metrics)  
    async with httpx.AsyncClient(http2=True) as client:  
        for _ in range(min(3, args.concurrency)):  
            await make_request(client, url, headers, payload, timeout)  
  
    metrics = Metrics()  
    metrics.configured_concurrency = args.concurrency  
  
    stop_event = asyncio.Event()  
    async with httpx.AsyncClient(http2=True) as client:  
        metrics.start_timer()  
        tasks = [  
            asyncio.create_task(worker(stop_event, metrics, client, url, headers, payload, timeout))  
            for _ in range(args.concurrency)  
        ]  
        await asyncio.sleep(args.duration)  
        stop_event.set()  
        await asyncio.gather(*tasks, return_exceptions=True)  
        metrics.end_timer()  
  
    return metrics.summary()  
  
def parse_args():  
    p = argparse.ArgumentParser(description="Benchmark /v1/chat/completions throughput")  
    p.add_argument("--concurrency", type=int, default=96)  
    p.add_argument("--duration", type=int, default=60, help="Benchmark active duration seconds")  
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")  
    p.add_argument("--max-tokens", type=int, default=200)  
    p.add_argument("--temperature", type=float, default=0.7)  
    p.add_argument("--prompt-file", type=str, default="sample_doc.md")  
    p.add_argument("--request-timeout", type=float, default=60.0)  
    return p.parse_args()  
  
def main():  
    args = parse_args()  
    start = time.time()  
    summary = asyncio.run(run_benchmark(args))  
    elapsed = time.time() - start  
    print("\n===== BENCHMARK SUMMARY =====")  
    print(json.dumps(summary, indent=2))  
    print(f"Total wall-clock elapsed (incl. warmup + teardown): {elapsed:.2f}s")  
  
if __name__ == "__main__":  
    main()  