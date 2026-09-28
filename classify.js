// A minimal vision classifier: one request, one forward pass, and a probability for every
// answer read from the logprobs. `image` is a data: URL; return { answer, probs }.

// The questions. Each one becomes a field of the JSON answer and a line of the prompt.
// A choice's options start with different letters: each is recognised by its first token.
const questions = {
  person: { type: 'boolean', true: 'a real person is visible', false: 'no real person is visible' },
  setting: { type: 'choice', options: { indoors: 'inside a building or vehicle', outdoors: 'outside', unclear: '' } },
  boat: { type: 'boolean', true: 'a boat is visible', false: 'no boat is visible' },
  car: { type: 'boolean', true: 'a car is visible', false: 'no car is visible' },
  animal: { type: 'boolean', true: 'an animal is visible', false: 'no animal is visible' },
  is_food: { type: 'boolean', true: 'food is visible', false: 'no food is visible' },
  subjects: { type: 'choice', options: { a_one: 'there is one distinct subject', b_two: 'there is two distinct subjects', c_three: 'there are three distinct subjects', d_four: 'there are four distinct subjects', e_more_than_four: 'there are five or more distinct subjects', f_zero: 'there are no distinct subjects' } },
};

// A true/false question is just a choice between 'true' and 'false'.
const options = (q) => (q.type === 'boolean' ? { true: q.true, false: q.false } : q.options);
const describe = ([name, about]) => (about ? `${name} (${about})` : name);
const prompt = 'Answer each question about the image, picking one of the listed values:\n' +
  Object.entries(questions).map(([key, q]) => `- ${key}: ${Object.entries(options(q)).map(describe).join(', ')}`).join('\n');

// All answered together as JSON the model is constrained to produce.
const schema = {
  type: 'object',
  properties: Object.fromEntries(Object.entries(questions).map(([key, q]) =>
    [key, q.type === 'boolean' ? { type: 'boolean' } : { enum: Object.keys(q.options) }])),
  required: Object.keys(questions),
};

const res = await fetch('/v1/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'gpt-4.1-mini',  // the demo server pins the model whatever this says
    temperature: 0,
    logprobs: true,     // return the log-probability of each output token...
    top_logprobs: 10,   // ...and of the 10 tokens the model weighed against it
    response_format: { type: 'json_schema', json_schema: { name: 'answer', schema } },
    messages: [{ role: 'user', content: [
      { type: 'image_url', image_url: { url: image } },
      { type: 'text', text: prompt },
    ] }],
  }),
});
if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
const answer = (await res.json()).choices[0];

// Walk the answer token by token. Where a value starts, the alternatives the model weighed
// at that one token ARE the probabilities of each possible answer — no extra requests.
const word = (t) => t.replace(/[\s"]/g, '');
const probs = {};
let text = '';
for (const { token, top_logprobs } of answer.logprobs.content) {
  const key = text.match(/"(\w+)":\s*"?$/)?.[1];  // the key this token is the value of
  if (questions[key] && !probs[key] && word(token)) {
    const choices = Object.keys(options(questions[key]));
    probs[key] = Object.fromEntries(choices.map((c) => [c, 0]));
    for (const alt of top_logprobs) {
      const w = word(alt.token);
      const c = w && choices.find((c) => c.startsWith(w));  // 'unc' is the first token of 'unclear'
      if (c) probs[key][c] += Math.exp(alt.logprob);
    }
  }
  text += token;
}

return { answer: JSON.parse(answer.message.content), probs };
