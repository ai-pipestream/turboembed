// Extract the pinned real model table without generating or repeating rows.
// node potion_search_data.mjs model.safetensors output-directory
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const [input, output] = process.argv.slice(2);
const bytes = fs.readFileSync(input);
const sha256 = b => crypto.createHash('sha256').update(b).digest('hex');
const expected = '99f6c33204c9231a7391871b7a3c91409b532c8f587a9ea44fc282303d8dec28';
if (sha256(bytes) !== expected) throw new Error('model digest differs from pinned recipe');
const length = Number(bytes.readBigUInt64LE(0));
const table = JSON.parse(bytes.subarray(8, 8 + length).toString()).embeddings;
if (table.dtype !== 'F32' || String(table.shape) !== '63091,512') throw new Error('unexpected embedding table');
const tensor = bytes.subarray(8 + length + table.data_offsets[0], 8 + length + table.data_offsets[1]);
if (tensor.length !== 63091 * 512 * 4) throw new Error('wrong tensor length');
fs.mkdirSync(output, {recursive: true});
fs.writeFileSync(path.join(output, 'vectors.f32'), tensor);
const source = {
  model: 'minishlab/potion-base-32M', license: 'MIT',
  commit: '1e5a03f8eeb2c98b928fbbd846f22f816360919f',
  url: 'https://huggingface.co/minishlab/potion-base-32M/resolve/1e5a03f8eeb2c98b928fbbd846f22f816360919f/model.safetensors',
  model_sha256: expected, tensor: 'embeddings', shape: table.shape,
  vectors_sha256: sha256(tensor),
};
fs.writeFileSync(path.join(output, 'source.json'), JSON.stringify(source, null, 2) + '\n');
console.log(JSON.stringify(source));
