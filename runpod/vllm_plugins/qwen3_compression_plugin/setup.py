"""
Qwen3 Compression vLLM Plugin for vLLM 0.14.x

Qwen3-0.6B based token classification model for context compression.
Install with: pip install -e .

NOTE: This uses a flat layout where setup.py lives IN the package directory.
find_packages() returns [] because there are no sub-packages.
The package is importable via PYTHONPATH=/app/vllm_plugins (set in Dockerfile).
"""

from setuptools import setup

setup(
    name="vllm-qwen3-compression-plugin",
    version="0.14.0",
    description="Qwen3 Token Classification model plugin for vLLM 0.14.x (compression service)",
    author="Latence Team",
    package_dir={"qwen3_compression_plugin": "."},
    packages=["qwen3_compression_plugin"],
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.40",
    ],
    # vLLM 0.14.x: must use vllm.general_plugins so the plugin
    # is loaded BEFORE ModelConfig validation checks architectures.
    entry_points={
        "vllm.general_plugins": [
            "qwen3_compression = qwen3_compression_plugin:register_model",
        ],
    },
)
