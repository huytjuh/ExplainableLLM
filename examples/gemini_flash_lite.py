import _path_setup  # noqa: F401

from S01_client import LLMClient, MissingAPIKeyError


if __name__ == "__main__":
    client = LLMClient.from_config("config/gemini.yaml")
    try:
        response = client.generate(
            "Explain tokenization in one sentence.",
            generation={"max_output_tokens": 120},
        )
        print(response.text)
    except MissingAPIKeyError as error:
        print(error)
