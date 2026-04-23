from setuptools import setup


setup(
    name="latence-trace-nli-plugin",
    version="0.1.0",
    description="latence-trace BYOP NLI plugin for vllm-factory",
    package_dir={"latence_trace_nli_plugin": "."},
    packages=["latence_trace_nli_plugin"],
    python_requires=">=3.11",
    install_requires=[
        "torch",
        "transformers",
        "vllm==0.19.0",
        "vllm-factory",
    ],
    entry_points={
        "vllm.general_plugins": [
            "nli_mdeberta = latence_trace_nli_plugin:register",
        ],
        "vllm.io_processor_plugins": [
            "nli_mdeberta = latence_trace_nli_plugin.io_processor:get_processor_cls",
        ],
    },
)
