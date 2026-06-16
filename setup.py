"""Fallback setup script for environments that do not fully use pyproject.toml."""

import os
from pathlib import Path

from setuptools import find_packages, setup

root = Path(__file__).parent

with open(os.path.join(root, "verl/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    "fastapi",
    "hydra-core",
    "numpy<2.0.0",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0",
    "torchdata",
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "transformers",
    "wandb",
    "packaging>=20.0",
    "tensorboard",
    "uvicorn",
]

extras_require = {
    "test": ["pytest", "pre-commit", "py-spy", "pytest-asyncio", "pytest-rerunfailures"],
    "gpu": ["liger-kernel", "flash-attn"],
    "math": ["math-verify"],
    "vllm": ["tensordict>=0.8.0,<=0.10.0,!=0.9.0", "vllm>=0.8.5,<=0.12.0"],
    "sglang": ["tensordict>=0.8.0,<=0.10.0,!=0.9.0", "sglang[srt,openai]==0.5.8", "torch==2.9.1"],
    "mcore": ["mbridge"],
    "trtllm": ["tensorrt-llm>=1.2.0rc6"],
}

setup(
    name="trajweave",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    url="https://github.com/OpenDCAI/TrajWeave",
    license="Apache-2.0",
    author="OpenDCAI",
    description="TrajWeave: a Multi-Agent LLM RL framework built on the VERL training backend",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": [
            "trainer/config/*.yaml",
            "trainer/config/*/*.yaml",
            "experimental/*/config/*.yaml",
        ],
    },
    include_package_data=True,
    long_description=(root / "README.md").read_text(),
    long_description_content_type="text/markdown",
)
