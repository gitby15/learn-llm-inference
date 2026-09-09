import asyncio
import time
from pathlib import Path

from guidellm.benchmark import (
    BenchmarkScenario,
    GenerativeConsoleBenchmarkerProgress,
    benchmark_generative_text,
)
from guidellm.utils.console import Console

batch_client = [1,2,4,8,16]

async def main():
    # 先启动项目服务：uv run tim-inf，默认监听 http://127.0.0.1:8888
    timestamp_suffix = str(int(time.time()))[-6:]
    output_dir = Path("benchmark/results") / timestamp_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = BenchmarkScenario.create(
        scenario=None,
        spec={
            "backend": {
                "kind": "openai_http",
                "target": "http://127.0.0.1:8888",
                "model": "openbmb/MiniCPM5-1B",
                "request_format": "/v1/chat/completions",
                "validate_backend": False,
                "stream": True,
            },
            "profile": {
                "kind": "concurrent",
                "streams": batch_client,
            },
            "constraints": [
                {"kind": "max_requests", "count": [item * 20 for item in batch_client]},
            ],
            "data": [
                {
                    "kind": "synthetic_text",
                    "prompt_tokens": 256,
                    "output_tokens": 512,
                }
            ],
            "outputs": [
                {"kind": "json", "path": str(output_dir / "results.json")},
                {"kind": "csv", "path": str(output_dir / "results.csv")},
                {"kind": "html", "path": str(output_dir / "results.html")},
            ],
        },
    )

    _, output_results = await benchmark_generative_text(
        args=scenario,
        progress=GenerativeConsoleBenchmarkerProgress(),
        console=Console(),
    )
    print(output_results)


if __name__ == "__main__":
    asyncio.run(main())
