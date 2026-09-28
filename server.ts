// image-classifier-demo: a page that shows off a one-request vision classifier.
//   GET  /                     the page
//   GET  /classify.js          the script the page's editor starts with
//   GET  /api/random-image     a random photo from picsum.photos (Unsplash) as a data: URL
//   GET  /api/usage            how much of each rate limit is used (the visitor's and the site's)
//   POST /v1/chat/completions  the vision API, with the server's key added (the key never
//                              reaches the browser) and the request pinned to what the demo needs
// Both API routes are rate limited per visitor (RATE_LIMIT_PER_MIN, each) and across the whole
// site (SITE_CLASSIFY_PER_MIN, SITE_CLASSIFY_PER_DAY, SITE_GENERATE_PER_MIN), and the proxy only
// forwards requests shaped like the demo's classifier (see checkRequest).
// Run: bun run server.ts  (reads OPENAI_API_KEY from .env; see the README for the rest)

const PICSUM = 'https://picsum.photos';
const MAX_BODY_BYTES = 2_000_000;
const MAX_TOKENS = 512;
const MAX_TEXT_CHARS = 4_000;
const MAX_SCHEMA_BYTES = 8_192;
const DAY_MS = 86_400_000;

type Fetcher = (url: string, init?: RequestInit) => Promise<Response>;
type Server = { requestIP(req: Request): { address: string } | null };

/** A sliding window per key: at most `limit` hits in any `windowMs` (one minute by default). */
export class RateLimiter {
  #hits = new Map<string, number[]>();
  constructor(readonly limit: number, readonly windowMs = 60_000, readonly now = () => Date.now()) {}

  /** Records a hit unless over the limit; otherwise says how many seconds until the next one is allowed. */
  take(key: string): { ok: true } | { ok: false; retryAfter: number } {
    const now = this.now();
    const recent = (this.#hits.get(key) ?? []).filter((t) => now - t < this.windowMs);
    if (recent.length >= this.limit) {
      this.#hits.set(key, recent);
      return { ok: false, retryAfter: Math.ceil((recent[0]! + this.windowMs - now) / 1000) };
    }
    recent.push(now);
    this.#hits.set(key, recent);
    if (this.#hits.size > 10_000) this.#sweep(now);
    return { ok: true };
  }

  /** How much of the window `key` has used, without recording a hit; retryAfter is 0 while a slot is free. */
  peek(key: string): { used: number; limit: number; retryAfter: number } {
    const now = this.now();
    const recent = (this.#hits.get(key) ?? []).filter((t) => now - t < this.windowMs);
    // The oldest hit that must age out before the next one is allowed.
    const blocking = recent.length >= this.limit ? recent[recent.length - this.limit] : undefined;
    const retryAfter = blocking === undefined ? 0 : Math.ceil((blocking + this.windowMs - now) / 1000);
    return { used: recent.length, limit: this.limit, retryAfter };
  }

  #sweep(now: number): void {
    for (const [key, times] of this.#hits) if (times.every((t) => now - t >= this.windowMs)) this.#hits.delete(key);
  }
}

export interface Options {
  /** The OpenAI API's base URL, https://api.openai.com/v1. */
  upstream: string;
  apiKey?: string;
  /** Every request uses this model, whatever the page's script asks for. */
  model: string;
  fetcher?: Fetcher;
  /** Generates and classifies each allowed per visitor per minute; 0 turns limiting off. */
  ratePerMinute?: number;
  /**
   * A header carrying the visitor's real IP, set by a trusted proxy in front (e.g.
   * cf-connecting-ip behind Cloudflare). Unset: the connection's address. Only set it when
   * every request comes through that proxy — otherwise a client can put any value in it.
   */
  clientIpHeader?: string;
  /** Caps across all visitors together; 0 turns each off. */
  site?: { classifyPerMinute?: number; classifyPerDay?: number; generatePerMinute?: number };
  now?: () => number;
}

/**
 * Pins a chat request to what the demo needs, so the server can't be used as a general
 * proxy for the key: one model, a short answer, no streaming, one choice.
 */
export function pinRequest(body: Record<string, unknown>, model: string): Record<string, unknown> {
  const { n: _n, stream: _stream, max_completion_tokens: _mct, ...rest } = body;
  const asked = typeof body.max_tokens === 'number' ? body.max_tokens : MAX_TOKENS;
  return { ...rest, model, max_tokens: Math.min(Math.max(1, asked), MAX_TOKENS), stream: false };
}

const ALLOWED_KEYS = ['model', 'messages', 'temperature', 'logprobs', 'top_logprobs', 'response_format', 'max_tokens'];
const DATA_IMAGE = /^data:image\/(png|jpeg|webp|gif);base64,[A-Za-z0-9+/]+={0,2}$/;

const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const extraKey = (o: Record<string, unknown>, allowed: string[]) => Object.keys(o).find((k) => !allowed.includes(k));

/**
 * Null when the body is the shape the demo's classifier sends — one user message holding one
 * image (a data: URL) and one text prompt, answered as JSON — otherwise why it isn't. Everything
 * else (tools, streaming, text-only chats, other formats) is refused, so the key can only be
 * spent on classifying an image.
 */
export function checkRequest(body: Record<string, unknown>): string | null {
  const extra = extraKey(body, ALLOWED_KEYS);
  if (extra) return `"${extra}" is not allowed: this demo only forwards ${ALLOWED_KEYS.join(', ')}`;

  const { messages } = body;
  if (!Array.isArray(messages) || messages.length !== 1) return 'messages must hold exactly one message';
  const [message] = messages;
  if (!isObject(message) || message.role !== 'user') return 'the message must have role "user"';
  if (extraKey(message, ['role', 'content'])) return 'the message may only have role and content';
  const { content } = message;
  if (!Array.isArray(content) || content.length !== 2) return 'the message content must be an array of one image_url part and one text part';
  const image = content.find((p) => isObject(p) && p.type === 'image_url');
  const text = content.find((p) => isObject(p) && p.type === 'text');
  if (!image || !text) return 'the message content must be an array of one image_url part and one text part';
  if (extraKey(image, ['type', 'image_url']) || !isObject(image.image_url) || extraKey(image.image_url, ['url', 'detail'])) {
    return 'the image_url part must be { type: "image_url", image_url: { url } }';
  }
  if (typeof image.image_url.url !== 'string' || !DATA_IMAGE.test(image.image_url.url)) {
    return 'the image must be a data: URL (data:image/png|jpeg|webp|gif;base64,...)';
  }
  if (extraKey(text, ['type', 'text']) || typeof text.text !== 'string') return 'the text part must be { type: "text", text }';
  if (text.text.length > MAX_TEXT_CHARS) return `the prompt is ${text.text.length} characters; at most ${MAX_TEXT_CHARS} are allowed`;

  const format = body.response_format;
  if (!isObject(format) || format.type !== 'json_schema' || extraKey(format, ['type', 'json_schema'])) {
    return 'response_format must be { type: "json_schema", json_schema: { name, schema } }';
  }
  const js = format.json_schema;
  if (!isObject(js) || !isObject(js.schema) || extraKey(js, ['name', 'schema', 'strict'])) {
    return 'response_format.json_schema must be { name, schema } with schema an object';
  }
  const schemaBytes = new TextEncoder().encode(JSON.stringify(js.schema)).length;
  if (schemaBytes > MAX_SCHEMA_BYTES) return `the JSON schema is ${schemaBytes} bytes; at most ${MAX_SCHEMA_BYTES} are allowed`;
  if (js.name !== undefined && (typeof js.name !== 'string' || js.name.length > 64)) return 'response_format.json_schema.name must be a short string';
  if (js.strict !== undefined && typeof js.strict !== 'boolean') return 'response_format.json_schema.strict must be a boolean';

  if (body.model !== undefined && typeof body.model !== 'string') return 'model must be a string';
  if (body.temperature !== undefined && !(typeof body.temperature === 'number' && body.temperature >= 0 && body.temperature <= 2)) {
    return 'temperature must be a number from 0 to 2';
  }
  if (body.logprobs !== undefined && typeof body.logprobs !== 'boolean') return 'logprobs must be true or false';
  const top = body.top_logprobs;
  if (top !== undefined && !(Number.isInteger(top) && (top as number) >= 0 && (top as number) <= 20)) {
    return 'top_logprobs must be an integer from 0 to 20';
  }
  if (body.max_tokens !== undefined && !(Number.isInteger(body.max_tokens) && (body.max_tokens as number) >= 1)) {
    return 'max_tokens must be a positive integer';
  }
  return null;
}

export function makeRoutes(opts: Options) {
  const fetcher = opts.fetcher ?? fetch;
  const perMinute = opts.ratePerMinute ?? 6;
  const limiters = {
    generate: new RateLimiter(perMinute, 60_000, opts.now),
    classify: new RateLimiter(perMinute, 60_000, opts.now),
  };
  // Across all visitors: one shared key per limiter, so every request counts toward the same window.
  const siteLimiters = {
    classifyPerMinute: new RateLimiter(opts.site?.classifyPerMinute ?? 30, 60_000, opts.now),
    classifyPerDay: new RateLimiter(opts.site?.classifyPerDay ?? 1000, DAY_MS, opts.now),
    generatePerMinute: new RateLimiter(opts.site?.generatePerMinute ?? 60, 60_000, opts.now),
  };
  const SITE = 'site';
  const siteChecks = {
    generate: [['generatePerMinute', 'minute']],
    // The minute first: under a flood it refuses most requests, so they don't eat the day's budget.
    classify: [['classifyPerMinute', 'minute'], ['classifyPerDay', 'day']],
  } as const;

  /** Who's asking: the trusted proxy's header when configured (and present), else the connection. */
  const visitor = (req: Request, server?: Server): string => {
    const forwarded = opts.clientIpHeader ? req.headers.get(opts.clientIpHeader)?.split(',')[0]?.trim() : '';
    return forwarded || server?.requestIP(req)?.address || 'unknown';
  };

  /** Null when allowed; otherwise the 429 to send. */
  const limited = (action: keyof typeof limiters, req: Request, server?: Server): Response | null => {
    if (perMinute <= 0) return null;
    const verdict = limiters[action].take(visitor(req, server));
    if (verdict.ok) return null;
    return Response.json(
      { error: `Rate limit: ${perMinute} per minute. Try again in ${verdict.retryAfter}s.` },
      { status: 429, headers: { 'Retry-After': String(verdict.retryAfter) } },
    );
  };

  /** Null when the whole site still has budget for this action; otherwise the 429 to send. */
  const siteLimited = (action: keyof typeof siteChecks): Response | null => {
    for (const [name, per] of siteChecks[action]) {
      const limiter = siteLimiters[name];
      if (limiter.limit <= 0) continue;
      const verdict = limiter.take(SITE);
      if (verdict.ok) continue;
      const what = action === 'classify' ? 'classifications' : 'new images';
      return Response.json(
        { error: `The demo is busy: the whole site is limited to ${limiter.limit} ${what} a ${per}, across all visitors. Try again in ${verdict.retryAfter}s.` },
        { status: 429, headers: { 'Retry-After': String(verdict.retryAfter) } },
      );
    }
    return null;
  };

  return {
    '/': new Response(Bun.file(`${import.meta.dir}/index.html`)),
    '/classify.js': new Response(Bun.file(`${import.meta.dir}/classify.js`)),

    // What's been used of each limit, for the page's header; null where a limit is off. Records nothing.
    '/api/usage': (req: Request, server: Server) => {
      const who = visitor(req, server);
      const site = (limiter: RateLimiter) => (limiter.limit > 0 ? limiter.peek(SITE) : null);
      return Response.json({
        visitor: {
          classify: perMinute > 0 ? limiters.classify.peek(who) : null,
          generate: perMinute > 0 ? limiters.generate.peek(who) : null,
        },
        site: {
          classifyPerMinute: site(siteLimiters.classifyPerMinute),
          classifyPerDay: site(siteLimiters.classifyPerDay),
          generatePerMinute: site(siteLimiters.generatePerMinute),
        },
      }, { headers: { 'Cache-Control': 'no-store' } });
    },

    '/api/random-image': async (req: Request, server: Server) => {
      const tooMany = limited('generate', req, server) ?? siteLimited('generate');
      if (tooMany) return tooMany;
      try {
        // A random photo redirects to its own URL, which carries the photo's id.
        const pick = await fetcher(`${PICSUM}/640/480`, { redirect: 'manual' });
        const location = pick.headers.get('location') ?? '';
        const id = location.match(/\/id\/(\d+)\//)?.[1];
        if (!id) throw new Error(`picsum: no photo id (HTTP ${pick.status})`);
        const [image, info] = await Promise.all([
          fetcher(new URL(location, PICSUM).href),
          fetcher(`${PICSUM}/id/${id}/info`),
        ]);
        if (!image.ok) throw new Error(`image: HTTP ${image.status}`);
        const { author, url } = info.ok ? ((await info.json()) as { author: string; url: string }) : { author: 'unknown', url: '' };
        const type = image.headers.get('content-type') ?? 'image/jpeg';
        const dataUrl = `data:${type};base64,${(await image.bytes()).toBase64()}`;
        return Response.json({ dataUrl, description: `Photo by ${author}`, credit: url, source: 'Unsplash' });
      } catch (err) {
        return Response.json({ error: (err as Error).message }, { status: 502 });
      }
    },

    '/v1/chat/completions': {
      POST: async (req: Request, server: Server) => {
        const tooMany = limited('classify', req, server);
        if (tooMany) return tooMany;
        if (!opts.apiKey) {
          return Response.json({ error: 'OPENAI_API_KEY is not set (put it in .env)' }, { status: 500 });
        }
        const raw = await req.text();
        if (raw.length > MAX_BODY_BYTES) {
          return Response.json({ error: `request body over ${MAX_BODY_BYTES} bytes` }, { status: 413 });
        }
        let body: unknown;
        try {
          body = JSON.parse(raw);
        } catch {
          return Response.json({ error: 'request body is not JSON' }, { status: 400 });
        }
        if (!body || typeof body !== 'object' || Array.isArray(body)) {
          return Response.json({ error: 'request body must be a JSON object' }, { status: 400 });
        }
        const wrong = checkRequest(body as Record<string, unknown>);
        if (wrong) return Response.json({ error: `Refused: ${wrong}.` }, { status: 400 });
        // Only now, for a request that would really be sent, does it count against the whole site.
        const siteBusy = siteLimited('classify');
        if (siteBusy) return siteBusy;
        let res: Response;
        try {
          res = await fetcher(`${opts.upstream}/chat/completions`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${opts.apiKey}`,
            },
            body: JSON.stringify(pinRequest(body as Record<string, unknown>, opts.model)),
            signal: AbortSignal.timeout(120_000),
          });
        } catch (err) {
          // Details go to the log, not the visitor.
          console.error('[classify] upstream failed:', (err as Error).message);
          const timedOut = (err as Error).name === 'TimeoutError';
          return Response.json(
            { error: timedOut ? 'The model took too long to answer. Try again.' : 'The model service is unreachable right now. Try again shortly.' },
            { status: timedOut ? 504 : 502 },
          );
        }
        return new Response(res.body, {
          status: res.status,
          headers: { 'Content-Type': res.headers.get('content-type') ?? 'application/json' },
        });
      },
    },
  };
}

if (import.meta.main) {
  const upstream = 'https://api.openai.com/v1';
  const model = process.env.OPENAI_MODEL ?? 'gpt-4.1-mini';
  const server = Bun.serve({
    port: Number(process.env.PORT ?? 3000),
    idleTimeout: 180,
    // Never Bun's debug error page: it shows file paths and source lines.
    development: false,
    error(err) {
      console.error('[server] unhandled error:', err);
      return Response.json({ error: 'Internal error' }, { status: 500 });
    },
    routes: makeRoutes({
      upstream,
      model,
      apiKey: process.env.OPENAI_API_KEY,
      ratePerMinute: Number(process.env.RATE_LIMIT_PER_MIN ?? 6),
      clientIpHeader: process.env.CLIENT_IP_HEADER || undefined,
      site: {
        classifyPerMinute: Number(process.env.SITE_CLASSIFY_PER_MIN ?? 30),
        classifyPerDay: Number(process.env.SITE_CLASSIFY_PER_DAY ?? 1000),
        generatePerMinute: Number(process.env.SITE_GENERATE_PER_MIN ?? 60),
      },
    }),
  });
  console.log(`image-classifier-demo on http://localhost:${server.port} → ${upstream} (${model}), `
    + `rate limit ${process.env.RATE_LIMIT_PER_MIN ?? 6}/min per visitor, visitor from ${process.env.CLIENT_IP_HEADER || 'the connection'}; `
    + `site-wide ${process.env.SITE_CLASSIFY_PER_MIN ?? 30} classifies/min, ${process.env.SITE_CLASSIFY_PER_DAY ?? 1000}/day, `
    + `${process.env.SITE_GENERATE_PER_MIN ?? 60} generates/min`);
}
