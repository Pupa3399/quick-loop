"""Single veRL external-library entry point for Ouro integrations."""

from importlib import import_module

for module_name in (
    "ouro_search.verl.model_hooks",
    "ouro_search.vllm_model.registry",
):
    import_module(module_name)
