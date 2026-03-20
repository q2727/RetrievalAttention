import re


def parse_model_size(model_name: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)B", model_name)
    if match is None:
        raise ValueError(f"Cannot parse model size from model name: {model_name}")
    return float(match.group(1))
