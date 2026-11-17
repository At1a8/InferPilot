// Structured answers for agents that only print text. A role call with a schema gets an output
// contract appended to its prompt; the last JSON object in the reply is parsed and checked against
// the schema subset inferpilot.js uses: type (one or a list), enum, properties, required, items.

export function outputContract(schema) {
  return '\n\n## Output contract\n'
    + 'Do the work first. Then, as the very last thing in your reply, print your result as one JSON '
    + 'object in a single ```json fenced block, with nothing after the closing fence. It must match '
    + 'this JSON Schema (fields beyond it are allowed):\n\n```json\n'
    + JSON.stringify(schema, null, 2)
    + '\n```\n';
}

// The JSON object a reply ends with: the last ```json block, else the last balanced {...} that parses.
export function extractJson(text) {
  const s = String(text || '');
  const fenced = [...s.matchAll(/```(?:json)?[ \t]*\n([\s\S]*?)```/g)];
  for (let i = fenced.length - 1; i >= 0; i--) {
    try {
      return JSON.parse(fenced[i][1]);
    } catch { /* try the one before */ }
  }
  for (let end = s.lastIndexOf('}'); end >= 0; end = s.lastIndexOf('}', end - 1)) {
    let depth = 0;
    for (let i = end; i >= 0; i--) {
      if (s[i] === '}') depth++;
      else if (s[i] === '{' && --depth === 0) {
        try {
          return JSON.parse(s.slice(i, end + 1));
        } catch {
          break;
        }
      }
    }
  }
  throw new Error('the reply holds no JSON object');
}

function typeOf(v) {
  if (v === null) return 'null';
  if (Array.isArray(v)) return 'array';
  if (typeof v === 'number') return Number.isInteger(v) ? 'integer' : 'number';
  return typeof v;
}

function check(v, schema, path, errors) {
  if (!schema || typeof schema !== 'object') return;
  if (schema.type) {
    const want = Array.isArray(schema.type) ? schema.type : [schema.type];
    const got = typeOf(v);
    if (!want.includes(got) && !(got === 'integer' && want.includes('number'))) {
      errors.push(`${path || 'value'}: ${got}, not ${want.join(' | ')}`);
      return;
    }
  }
  if (schema.enum && !schema.enum.some((e) => JSON.stringify(e) === JSON.stringify(v))) {
    errors.push(`${path || 'value'}: ${JSON.stringify(v)} is not one of ${JSON.stringify(schema.enum)}`);
  }
  if (typeOf(v) === 'object') {
    for (const k of schema.required || []) {
      if (!(k in v)) errors.push(`${path ? `${path}.` : ''}${k}: missing`);
    }
    for (const [k, sub] of Object.entries(schema.properties || {})) {
      if (k in v) check(v[k], sub, path ? `${path}.${k}` : k, errors);
    }
  }
  if (typeOf(v) === 'array' && schema.items) v.forEach((item, i) => check(item, schema.items, `${path}[${i}]`, errors));
}

export function validate(value, schema) {
  const errors = [];
  check(value, schema, '', errors);
  return errors;
}
