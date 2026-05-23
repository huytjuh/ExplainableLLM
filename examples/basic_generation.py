import _path_setup  # noqa: F401

from inference import generate_tokens


ID_TO_TOKEN = {0: "LLMs", 1: "explain", 2: "tokens", 3: "."}


def next_logits(generated_ids: list[int]) -> list[float]:
    script = [
        [4.0, 1.0, 0.5, 0.0],
        [0.2, 4.0, 1.0, 0.0],
        [0.0, 0.5, 4.0, 0.1],
        [0.0, 0.1, 0.2, 4.0],
    ]
    return script[min(len(generated_ids), len(script) - 1)]


if __name__ == "__main__":
    steps = generate_tokens(next_logits, ID_TO_TOKEN, max_tokens=4, temperature=0.0)
    print(" ".join(step.selected_token for step in steps))
    for step in steps:
        print(step)
