## Getting started

```bash
$ pip install -r requirements.txt
$ export OPENAI_API_KEY=sk-...
$ python3 -c'from api import prompt; prompt("How can I turn DEBUG log level on?")'
 To turn DEBUG log level on, you can pass the command-line option `--log-level` with the value `DEBUG` when launching a Meilisearch instance.
SOURCES: https://github.com/meilisearch/documentation/blob/a4226f6ffdab4a124a4137098a6b510e0b00206e/learn/configuration/instance_options.md
```

Inspired by https://dagster.io/blog/chatgpt-langchain