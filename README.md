# image-classifier-demo

A small page that shows off a vision classifier built on **one request and one
forward pass**: the model answers every question at once as JSON, and the
probability of *every possible answer* is read from the logprobs at the token
where each value starts.

- **Left:** the classifier script, in an editor. Classify runs whatever is there,
  so you can change the questions and see the effect. **Add question** adds a
  true/false or pick-one question to it.
- **Right:** a random photo ([picsum.photos](https://picsum.photos), from
  Unsplash), **Generate image** / **Classify image**, and a probability bar for
  each option of each question.

The script is plain browser JavaScript (`classify.js`). The server (`server.ts`,
Bun, no dependencies) serves the page, fetches photos, and forwards the one API
call with the key added.

## Run

Requires [Bun](https://bun.sh) and an [OpenAI API key](https://platform.openai.com/api-keys).
The model must take images and return `logprobs` with a JSON-schema
`response_format` (e.g. `gpt-4.1-mini`, `gpt-4o`; not the reasoning models).

```bash
echo 'OPENAI_API_KEY=sk-…' > .env
bun run server.ts        # http://localhost:3000
bun test
```

| Variable | Default | |
|---|---|---|
| `OPENAI_API_KEY` | — | Your OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-4.1-mini` | The model every request uses |
| `RATE_LIMIT_PER_MIN` | `6` | Generates and classifies each allowed per visitor per minute (`0` turns it off) |
| `SITE_CLASSIFY_PER_MIN` | `30` | Classifies per minute across all visitors together (`0` turns it off) |
| `SITE_CLASSIFY_PER_DAY` | `1000` | Classifies per day across all visitors together (`0` turns it off) |
| `SITE_GENERATE_PER_MIN` | `60` | Generates per minute across all visitors, to be polite to picsum.photos (`0` turns it off) |
| `CLIENT_IP_HEADER` | — | Header with the visitor's real IP from a trusted proxy, e.g. `cf-connecting-ip` behind Cloudflare. Unset: the connection's address. Only set it when every request comes through that proxy, or clients can spoof it |
| `PORT` | `3000` | |

Or with Docker: `docker build -t image-classifier-demo . && docker run --env-file .env -p 3000:3000 image-classifier-demo`.

## Security

The server spends **your** API key on every Classify, so it limits what the key
can be used for and how often:

- **Request shape.** The proxy only forwards what the classifier sends: one
  user message holding one `data:` image and one text prompt (at most 4,000
  characters), a `json_schema` response format (schema at most 8 KB), and
  `model`, `temperature`, `logprobs`, `top_logprobs`, `max_tokens`. Anything
  else (tools, streaming, text-only chat, other fields) gets a 400 saying why.
  On top, every request is pinned to one model, at most 512 output tokens and
  one choice, and bodies over 2 MB are refused.
- **Per visitor.** 6 generates and 6 classifies a minute by default. Set
  `CLIENT_IP_HEADER` behind a proxy, or every visitor shares one address.
- **Site-wide.** 30 classifies a minute and 1,000 a day across everyone, and 60
  generates a minute, so many visitors (or IPs) can't multiply the per-visitor
  limit without bound. Over it, the page is told the whole site is busy.
  The page's header shows both, from `GET /api/usage`.

The limits live in memory, per process: they reset on restart and aren't
shared between replicas. Give the demo **its own OpenAI project and key**,
with a monthly budget and rate limits set on the project and only the one model
allowed, so the worst case is bounded by OpenAI too, and the key can be revoked
without touching anything else. Keep `.env` out of git.

## License

MIT
