from setuptools import setup


setup(
    name="latence-trace-minicheck-t5-plugin",
    version="0.1.0",
    description=(
        "latence-trace BYOP plugin: MiniCheck-Flan-T5-Large served via "
        "vllm-factory's Triton-optimised MT5 encoder + a single-step "
        "decoder for grounded NLI."
    ),
    # Top-level package name intentionally matches the entry-point name
    # so :func:`runpod.handler._minicheck_plugin_available` (which uses
    # ``importlib.util.find_spec("minicheck_t5")``) can detect the
    # plugin without importing it. Same convention as
    # ``moderncolbert_batched``.
    package_dir={"minicheck_t5": "."},
    packages=["minicheck_t5"],
    python_requires=">=3.11",
    install_requires=[
        "torch",
        "transformers",
        "vllm==0.19.0",
        "vllm-factory",
    ],
    entry_points={
        # Runs at worker startup so the model + config are registered
        # with vLLM's ModelRegistry before the engine builds the model.
        "vllm.general_plugins": [
            "minicheck_t5 = minicheck_t5:register",
        ],
        # IO processor that handles the (premise, hypothesis) →
        # entailment-probability translation on the wire.
        "vllm.io_processor_plugins": [
            "minicheck_t5_io = minicheck_t5.io_processor:get_processor_cls",
        ],
    },
)
