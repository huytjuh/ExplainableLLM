import _path_setup  # noqa: F401

from gemini_flash_lite import GeminiFlashLiteClient


if __name__ == "__main__":
    client = GeminiFlashLiteClient()
    response = client.generate("Explain tokenization in one sentence.")
    print(response.text)
