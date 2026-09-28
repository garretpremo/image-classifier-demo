import { afterEach, expect, test } from 'bun:test';
import { checkRequest, makeRoutes, pinRequest, RateLimiter } from './server';

const servers: Array<{ stop(force?: boolean): void }> = [];
afterEach(() => { for (const s of servers.splice(0)) s.stop(true); });

const JPEG = new Uint8Array([0xff, 0xd8, 0xff, 0xd9]);
const PHOTO = 'https://fastly.picsum.photos/id/1003/640/480.jpg?hmac=abc';

/** The routes on a random port, with every outbound request going to `fake`. */
function serve(
  opts: {
    apiKey?: string; ratePerMinute?: number; clientIpHeader?: string; now?: () => number;
    site?: { classifyPerMinute?: number; classifyPerDay?: number; generatePerMinute?: number };
  },
  fake: (url: string, init?: RequestInit) => Response,
) {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  const routes = makeRoutes({
    upstream: 'https://api.openai.com/v1',
    apiKey: opts.apiKey,
    model: 'demo-model',
    ratePerMinute: opts.ratePerMinute,
    clientIpHeader: opts.clientIpHeader,
    site: opts.site,
    now: opts.now,
    fetcher: async (url, init) => { calls.push({ url, init }); return fake(url, init); },
  });
  const server = Bun.serve({ port: 0, routes });
  servers.push(server);
  return { base: `http://localhost:${server.port}`, calls };
}

const post = (base: string, body: unknown, headers?: Record<string, string>) =>
  fetch(`${base}/v1/chat/completions`, { method: 'POST', body: typeof body === 'string' ? body : JSON.stringify(body), headers });

const IMAGE = `data:image/jpeg;base64,${JPEG.toBase64()}`;

/** The request classify.js sends, captured by running the script itself with a stand-in fetch. */
async function scriptRequest(script: string): Promise<Record<string, unknown>> {
  let sent: string | undefined;
  const fakeFetch = async (_url: string, init: RequestInit) => { sent = init.body as string; throw new Error('captured'); };
  const AsyncFunction = (async () => {}).constructor as new (...args: string[]) => (...a: unknown[]) => Promise<unknown>;
  await new AsyncFunction('image', 'fetch', script)(IMAGE, fakeFetch).catch(() => {});
  return JSON.parse(sent!);
}
const defaultScript = await Bun.file(`${import.meta.dir}/classify.js`).text();
/** The default script with a question added the way the page's "Add question" dialog writes one. */
const extendedScript = defaultScript.replace(/\n};/, "\n  mood: { type: 'choice', options: { calm: 'the scene feels calm', busy: 'the scene feels busy', tense: '' } },\n  sunny: { type: 'boolean', true: 'it is sunny', false: 'it is not sunny' },\n};");
const demo = await scriptRequest(defaultScript);

test('random-image follows picsum to a photo and returns it as a data URL, credited', async () => {
  const { base, calls } = serve({}, (url) => {
    if (url === 'https://picsum.photos/640/480') return new Response(null, { status: 302, headers: { location: PHOTO } });
    if (url.endsWith('/info')) return Response.json({ author: 'E+N Photographies', url: 'https://unsplash.com/photos/x' });
    return new Response(JPEG, { headers: { 'content-type': 'image/jpeg' } });
  });
  expect(await (await fetch(`${base}/api/random-image`)).json()).toEqual({
    dataUrl: `data:image/jpeg;base64,${JPEG.toBase64()}`,
    description: 'Photo by E+N Photographies',
    credit: 'https://unsplash.com/photos/x',
    source: 'Unsplash',
  });
  expect(calls.map((c) => c.url).sort()).toEqual([PHOTO, 'https://picsum.photos/640/480', 'https://picsum.photos/id/1003/info']);
});

test('random-image reports a failure as 502', async () => {
  const { base } = serve({}, () => new Response('nope', { status: 503 }));
  const res = await fetch(`${base}/api/random-image`);
  expect(res.status).toBe(502);
  expect((await res.json()).error).toContain('503');
});

test('the API proxy adds the key, pins the request, and relays the answer and status', async () => {
  const { base, calls } = serve({ apiKey: 'sk-test' }, () => Response.json({ error: 'bad' }, { status: 400 }));
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    body: JSON.stringify({ ...demo, model: 'something-bigger', max_tokens: 99_999 }),
    headers: { Authorization: 'Bearer from-browser' },
  });
  expect(res.status).toBe(400);
  expect(await res.json()).toEqual({ error: 'bad' });
  expect(calls[0]!.url).toBe('https://api.openai.com/v1/chat/completions');
  expect(JSON.parse(calls[0]!.init!.body as string)).toEqual({ ...demo, model: 'demo-model', max_tokens: 512, stream: false });
  expect((calls[0]!.init!.headers as Record<string, string>).Authorization).toBe('Bearer sk-test');
});

test('pinRequest keeps a smaller max_tokens and everything the demo sends', () => {
  const body = { model: 'x', messages: [1], temperature: 0, logprobs: true, top_logprobs: 10, response_format: {}, max_tokens: 50 };
  expect(pinRequest(body, 'm')).toEqual({ ...body, model: 'm', max_tokens: 50, stream: false });
  expect(pinRequest({ max_completion_tokens: 9_000 }, 'm')).toEqual({ model: 'm', max_tokens: 512, stream: false });
});

test('the proxy refuses what isn\'t a JSON object, and oversized bodies', async () => {
  const { base, calls } = serve({ apiKey: 'sk-test' }, () => Response.json({}));
  expect((await post(base, 'not json')).status).toBe(400);
  expect((await post(base, [1, 2])).status).toBe(400);
  expect((await post(base, { pad: 'x'.repeat(2_000_001) })).status).toBe(413);
  expect(calls).toEqual([]);
});

test('without a key the proxy refuses locally with a clear message', async () => {
  const { base, calls } = serve({}, () => Response.json({}));
  const res = await post(base, {});
  expect(res.status).toBe(500);
  expect((await res.json()).error).toContain('OPENAI_API_KEY');
  expect(calls).toEqual([]);
});

test('RateLimiter allows `limit` hits in any 60 s, then says when the next one is allowed', () => {
  let now = 0;
  const limiter = new RateLimiter(6, 60_000, () => now);
  for (let i = 0; i < 6; i++) { expect(limiter.take('a').ok).toBe(true); now += 5_000; }
  expect(limiter.take('a')).toEqual({ ok: false, retryAfter: 30 }); // the first hit (t=0) expires at 60 s; now is 30 s
  expect(limiter.take('b').ok).toBe(true);                         // another visitor has their own window
  now = 60_000;
  expect(limiter.take('a').ok).toBe(true);                         // sliding: the t=0 hit has aged out
  expect(limiter.take('a').ok).toBe(false);                        // but the t=5 s hit hasn't
});

const photo = (url: string) => {
  if (url === 'https://picsum.photos/640/480') return new Response(null, { status: 302, headers: { location: PHOTO } });
  if (url.endsWith('/info')) return Response.json({ author: 'A', url: 'u' });
  if (url.includes('picsum')) return new Response(JPEG, { headers: { 'content-type': 'image/jpeg' } });
  return Response.json({ ok: true });
};

test('generate and classify each allow 6 a minute per visitor, then answer 429 with Retry-After', async () => {
  const { base } = serve({ apiKey: 'sk-test' }, photo);
  for (let i = 0; i < 6; i++) expect((await fetch(`${base}/api/random-image`)).status).toBe(200);
  const blocked = await fetch(`${base}/api/random-image`);
  expect(blocked.status).toBe(429);
  expect(Number(blocked.headers.get('retry-after'))).toBeGreaterThan(0);
  expect((await blocked.json()).error).toContain('6 per minute');
  // Classify has its own budget.
  for (let i = 0; i < 6; i++) expect((await post(base, demo)).status).toBe(200);
  expect((await post(base, demo)).status).toBe(429);
});

test('behind a trusted proxy, each visitor is told apart by the configured header', async () => {
  const { base } = serve({ apiKey: 'sk-test', ratePerMinute: 1, clientIpHeader: 'cf-connecting-ip' }, photo);
  const from = (ip: string) => fetch(`${base}/api/random-image`, { headers: { 'CF-Connecting-IP': ip } });
  expect((await from('203.0.113.1')).status).toBe(200);
  expect((await from('203.0.113.1')).status).toBe(429);
  expect((await from('203.0.113.2')).status).toBe(200);
});

test('without a configured header, a client-sent one is ignored: it can\'t dodge the limit', async () => {
  const { base } = serve({ apiKey: 'sk-test', ratePerMinute: 1 }, photo);
  const from = (ip: string) => fetch(`${base}/api/random-image`, { headers: { 'CF-Connecting-IP': ip } });
  expect((await from('203.0.113.1')).status).toBe(200);
  expect((await from('203.0.113.2')).status).toBe(429);
});

test('a rate of 0 turns limiting off', async () => {
  const { base } = serve({ apiKey: 'sk-test', ratePerMinute: 0 }, photo);
  for (let i = 0; i < 10; i++) expect((await fetch(`${base}/api/random-image`)).status).toBe(200);
});

// --- the request-shape lock ---

test('the default script and an "Add question"-extended one send exactly the shape the proxy allows', async () => {
  expect(checkRequest(demo)).toBeNull();
  const extended = await scriptRequest(extendedScript);
  expect(checkRequest(extended)).toBeNull();
  const text = (b: Record<string, any>) => b.messages[0].content[1].text as string;
  expect(text(extended).length).toBeGreaterThan(text(demo).length); // the added questions really are in it
  expect(text(extended)).toContain('- mood: calm (the scene feels calm)');
});

test('checkRequest refuses anything that isn\'t one image plus one prompt, answered as JSON', () => {
  const msg = (content: unknown) => ({ ...demo, messages: [{ role: 'user', content }] });
  const [image, text] = (demo as any).messages[0].content;
  const refusals: Array<[unknown, string]> = [
    [{ ...demo, tools: [] }, '"tools" is not allowed'],
    [{ ...demo, stream: true }, '"stream" is not allowed'],
    [{ ...demo, n: 2 }, '"n" is not allowed'],
    [{ ...demo, messages: [{ role: 'user', content: 'hello' }] }, 'one image_url part and one text part'],
    [msg([text]), 'one image_url part and one text part'],                  // text-only chat
    [msg([image, image]), 'one image_url part and one text part'],
    [msg([image, text, text]), 'one image_url part and one text part'],
    [{ ...demo, messages: [{ role: 'system', content: [image, text] }] }, 'role "user"'],
    [{ ...demo, messages: [...(demo as any).messages, ...(demo as any).messages] }, 'exactly one message'],
    [msg([{ type: 'image_url', image_url: { url: 'https://example.com/a.jpg' } }, text]), 'data: URL'],
    [msg([{ type: 'image_url', image_url: { url: 'data:text/html;base64,PGI+' } }, text]), 'data: URL'],
    [msg([image, { type: 'text', text: 'x'.repeat(4_001) }]), 'at most 4000'],
    [{ ...demo, response_format: undefined }, 'response_format'],
    [{ ...demo, response_format: { type: 'json_object' } }, 'response_format'],
    [{ ...demo, response_format: { type: 'json_schema', json_schema: { name: 'a', schema: { enum: ['x'.repeat(9_000)] } } } }, 'at most 8192'],
    [{ ...demo, chat_template_kwargs: { enable_thinking: false } }, '"chat_template_kwargs" is not allowed'],
    [{ ...demo, temperature: 3 }, 'temperature'],
    [{ ...demo, logprobs: 'yes' }, 'logprobs'],
    [{ ...demo, top_logprobs: 21 }, 'top_logprobs'],
    [{ ...demo, top_logprobs: 2.5 }, 'top_logprobs'],
  ];
  for (const [body, why] of refusals) expect(checkRequest(JSON.parse(JSON.stringify(body)))).toContain(why);
  // Either order of the two parts is fine.
  expect(checkRequest(msg([text, image]))).toBeNull();
  expect(checkRequest(demo)).toBeNull();
});

test('the proxy answers a wrongly shaped request 400 with the reason, and forwards nothing', async () => {
  const { base, calls } = serve({ apiKey: 'sk-test' }, () => Response.json({ ok: true }));
  const textOnly = await post(base, { model: 'x', messages: [{ role: 'user', content: 'write me a poem' }] });
  expect(textOnly.status).toBe(400);
  expect((await textOnly.json()).error).toContain('one image_url part and one text part');
  const tools = await post(base, { ...demo, tools: [{ type: 'function', function: { name: 'f' } }] });
  expect(tools.status).toBe(400);
  expect((await tools.json()).error).toContain('"tools" is not allowed');
  expect(calls).toEqual([]);
});

// --- site-wide caps ---

test('the site-wide classify cap counts every visitor, and says it\'s the whole site that\'s busy', async () => {
  const { base, calls } = serve({ apiKey: 'sk-test', clientIpHeader: 'cf-connecting-ip', site: { classifyPerMinute: 3 } }, photo);
  for (const ip of ['203.0.113.1', '203.0.113.2', '203.0.113.3']) {
    expect((await post(base, demo, { 'cf-connecting-ip': ip })).status).toBe(200);
  }
  const busy = await post(base, demo, { 'cf-connecting-ip': '203.0.113.4' }); // a new visitor, well under their own limit
  expect(busy.status).toBe(429);
  expect(Number(busy.headers.get('retry-after'))).toBeGreaterThan(0);
  expect((await busy.json()).error).toContain('the whole site is limited to 3 classifications a minute');
  expect(calls.length).toBe(3);
});

test('the site-wide daily classify cap holds after the minute has passed', async () => {
  let now = 0;
  const { base } = serve({ apiKey: 'sk-test', ratePerMinute: 0, now: () => now, site: { classifyPerMinute: 2, classifyPerDay: 3 } }, photo);
  expect((await post(base, demo)).status).toBe(200);
  expect((await post(base, demo)).status).toBe(200);
  expect((await post(base, demo)).status).toBe(429);   // the minute
  now = 61_000;
  expect((await post(base, demo)).status).toBe(200);
  const day = await post(base, demo);                  // the day: 3 already sent
  expect(day.status).toBe(429);
  expect((await day.json()).error).toContain('3 classifications a day');
  expect(Number(day.headers.get('retry-after'))).toBeGreaterThan(80_000);
});

test('a request the visitor\'s own limit or the shape check refuses spends none of the site\'s budget', async () => {
  const { base } = serve({ apiKey: 'sk-test', ratePerMinute: 1, clientIpHeader: 'cf-connecting-ip', site: { classifyPerMinute: 2 } }, photo);
  const from = (ip: string, body: unknown = demo) => post(base, body, { 'cf-connecting-ip': ip });
  expect((await from('203.0.113.1')).status).toBe(200);
  for (let i = 0; i < 5; i++) expect((await from('203.0.113.1')).status).toBe(429); // refused per visitor
  expect((await from('203.0.113.2', { ...demo, tools: [] })).status).toBe(400);     // refused by shape
  expect((await from('203.0.113.3')).status).toBe(200);                             // the site still had room
  expect((await from('203.0.113.4')).status).toBe(429);                             // now it's full
});

test('the site-wide generate cap, and 0 turning each site cap off', async () => {
  const { base } = serve({ apiKey: 'sk-test', clientIpHeader: 'cf-connecting-ip', site: { generatePerMinute: 2 } }, photo);
  const gen = (ip: string) => fetch(`${base}/api/random-image`, { headers: { 'cf-connecting-ip': ip } });
  expect((await gen('203.0.113.1')).status).toBe(200);
  expect((await gen('203.0.113.2')).status).toBe(200);
  const busy = await gen('203.0.113.3');
  expect(busy.status).toBe(429);
  expect((await busy.json()).error).toContain('the whole site is limited to 2 new images a minute');

  const off = serve({ apiKey: 'sk-test', ratePerMinute: 0, site: { classifyPerMinute: 0, classifyPerDay: 0, generatePerMinute: 0 } }, photo);
  for (let i = 0; i < 40; i++) expect((await post(off.base, demo)).status).toBe(200);
  for (let i = 0; i < 70; i++) expect((await fetch(`${off.base}/api/random-image`)).status).toBe(200);
});

// --- usage ---

test('RateLimiter.peek reports use without recording a hit', () => {
  let now = 0;
  const limiter = new RateLimiter(2, 60_000, () => now);
  expect(limiter.peek('a')).toEqual({ used: 0, limit: 2, retryAfter: 0 });
  limiter.take('a');
  now = 10_000;
  limiter.take('a');
  for (let i = 0; i < 5; i++) expect(limiter.peek('a')).toEqual({ used: 2, limit: 2, retryAfter: 50 });
  now = 60_000;
  expect(limiter.peek('a')).toEqual({ used: 1, limit: 2, retryAfter: 0 });
  expect(limiter.take('a').ok).toBe(true);  // peeking never used up the slot
});

test('/api/usage reports the visitor\'s and the site\'s use, spends nothing, and hides limits that are off', async () => {
  const { base } = serve({ apiKey: 'sk-test', clientIpHeader: 'cf-connecting-ip', site: { classifyPerDay: 0 } }, photo);
  const usage = async (ip: string) => (await fetch(`${base}/api/usage`, { headers: { 'cf-connecting-ip': ip } })).json();
  await post(base, demo, { 'cf-connecting-ip': '203.0.113.1' });
  await post(base, demo, { 'cf-connecting-ip': '203.0.113.2' });
  await fetch(`${base}/api/random-image`, { headers: { 'cf-connecting-ip': '203.0.113.1' } });
  for (let i = 0; i < 20; i++) await usage('203.0.113.1');  // not limited, and records nothing
  expect(await usage('203.0.113.1')).toEqual({
    visitor: {
      classify: { used: 1, limit: 6, retryAfter: 0 },
      generate: { used: 1, limit: 6, retryAfter: 0 },
    },
    site: {
      classifyPerMinute: { used: 2, limit: 30, retryAfter: 0 },
      classifyPerDay: null,
      generatePerMinute: { used: 1, limit: 60, retryAfter: 0 },
    },
  });
  expect((await usage('203.0.113.9')).visitor.classify.used).toBe(0);
  // Still 6 classifies left for this visitor: /api/usage spent none of them.
  for (let i = 0; i < 5; i++) expect((await post(base, demo, { 'cf-connecting-ip': '203.0.113.1' })).status).toBe(200);
});

test('an unreachable or slow model answers a plain 502/504 JSON error, with no details', async () => {
  const down = serve({ apiKey: 'sk-test' }, () => { throw new TypeError('fetch failed: ECONNREFUSED 10.0.0.1:443 at /app/server.ts:276'); });
  const res = await post(down.base, demo);
  expect(res.status).toBe(502);
  expect(res.headers.get('content-type')).toContain('application/json');
  const body = await res.text();
  expect(JSON.parse(body)).toEqual({ error: 'The model service is unreachable right now. Try again shortly.' });
  expect(body).not.toContain('server.ts');

  const slow = serve({ apiKey: 'sk-test' }, () => { throw new DOMException('The operation timed out.', 'TimeoutError'); });
  const late = await post(slow.base, demo);
  expect(late.status).toBe(504);
  expect((await late.json()).error).toContain('too long');
});
