# Configuration

Use the committed provider YAML files as templates:

- `providers/gemini.yaml`
- `providers/chatgpt.yaml`
- `providers/claude.yaml`
- `providers/ollama.yaml`

Create a local file such as `gemini.local.yaml` for machine-specific settings. Keep API keys out of YAML and set them as environment variables:

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OLLAMA_HOST`

Example:

```python
from S01_client import LLMClient

client = LLMClient.from_config("config/providers/gemini.local.yaml")
response = client.generate("Explain RAG in one sentence.")
```
