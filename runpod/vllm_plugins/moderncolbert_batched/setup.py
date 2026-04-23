from setuptools import setup


setup(
    name="moderncolbert-batched",
    version="0.1.0",
    description=(
        "Vendored ModernColBERT IO processor with batched-input support "
        "(moderncolbert_batched_io)."
    ),
    package_dir={"moderncolbert_batched": "."},
    packages=["moderncolbert_batched"],
    python_requires=">=3.11",
    install_requires=[
        "torch",
        "transformers",
        "vllm-factory",
    ],
    entry_points={
        "vllm.io_processor_plugins": [
            "moderncolbert_batched_io = moderncolbert_batched.io_processor:get_processor_cls",
        ],
    },
)
