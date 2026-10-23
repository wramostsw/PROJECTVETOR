// tools/bancada_server.js — servidor local da bancada: serve a página e roda o cnx_cli a cada imagem enviada.
//
//   node tools\bancada_server.js [--porta 8765] [--exe caminho\cnx_cli.exe]
//
// O motor é C++; o navegador não o executa. Esta ponte recebe a imagem por HTTP, chama o cnx_cli de verdade
// (nenhuma vetorização acontece em JavaScript) e devolve o SVG com o log das etapas.
// Só escuta em 127.0.0.1 e só aceita Host/Origin locais. Uma vetorização por vez.
'use strict';
const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const argv = process.argv.slice(2);
const opt = (name, def) => { const i = argv.indexOf(name); return i >= 0 && argv[i + 1] ? argv[i + 1] : def; };
const PORT = parseInt(opt('--porta', '8765'), 10);
const ROOT = path.resolve(__dirname, '..');
const PAGE = path.join(__dirname, 'bancada-cnx.html');
const MAX_BODY = 60 * 1024 * 1024;
const TIMEOUT_MS = 10 * 60 * 1000;

function findExe() {
  const given = opt('--exe', process.env.CNX_CLI);
  const candidates = given ? [given] : ['build-bancada', 'build', path.join('build', 'Release'), 'build-release'].map((d) => path.join(ROOT, d, 'cnx_cli.exe'));
  return candidates.find((p) => fs.existsSync(p));
}
const EXE = findExe();
if (!EXE) {
  console.error('cnx_cli.exe não encontrado. Compile o alvo cnx_cli (pasta build-bancada) ou passe --exe caminho.');
  process.exit(1);
}

// Argumentos aceitos: só o que o cnx_cli entende, sem passar nada por shell.
function buildArgs(q) {
  const out = [];
  const perfil = q.get('perfil');
  if (perfil) { if (!/^[0-9]$/.test(perfil)) throw new Error('perfil inválido'); out.push('--perfil', perfil); }
  const cores = q.get('cores');
  if (cores) { if (!/^[0-9]{1,3}$/.test(cores) || +cores < 1) throw new Error('cores inválido'); out.push('--cores', cores); }
  const limiar = q.get('limiar');
  if (limiar) { if (!/^[0-9]+(\.[0-9]+)?$/.test(limiar) || +limiar <= 0) throw new Error('limiar inválido'); out.push('--limiar', limiar); }
  for (const line of q.getAll('set')) {
    const m = line.trim().match(/^([A-Za-z0-9_]+::[A-Za-z0-9_]+)\s+([-+0-9.A-Za-z_]+)$/);
    if (!m) throw new Error('parâmetro extra inválido: ' + line);
    out.push('--set', m[1], m[2]);
  }
  return out;
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []; let n = 0;
    req.on('data', (c) => { n += c.length; if (n > MAX_BODY) { reject(new Error('imagem grande demais (limite 60 MB)')); req.destroy(); } else chunks.push(c); });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}
const send = (res, code, obj) => { const b = Buffer.from(JSON.stringify(obj)); res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8', 'Content-Length': b.length, 'Cache-Control': 'no-store' }); res.end(b); };

let busy = false;
async function vectorize(req, res, url) {
  if (busy) return send(res, 429, { ok: false, error: 'já existe uma vetorização em andamento' });
  let args;
  try { args = buildArgs(url.searchParams); } catch (e) { return send(res, 400, { ok: false, error: e.message }); }
  let body;
  try { body = await readBody(req); } catch (e) { return send(res, 413, { ok: false, error: e.message }); }
  const ext = body.length > 8 && body[0] === 0x89 && body[1] === 0x50 && body[2] === 0x4e && body[3] === 0x47 ? '.png'
    : body.length > 3 && body[0] === 0xff && body[1] === 0xd8 ? '.jpg' : null;
  if (!ext) return send(res, 415, { ok: false, error: 'o motor lê PNG e JPEG' });
  busy = true;
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'cnx-bancada-'));
  const input = path.join(dir, 'entrada' + ext), output = path.join(dir, 'saida.svg');
  try {
    fs.writeFileSync(input, body);
    const t0 = Date.now(); let log = '';
    const code = await new Promise((resolve) => {
      const p = spawn(EXE, [...args, input, output], { cwd: dir, windowsHide: true });
      const timer = setTimeout(() => { log += '\n[tempo esgotado: 10 min]'; p.kill(); }, TIMEOUT_MS);
      p.stdout.on('data', (d) => { log += d; }); p.stderr.on('data', (d) => { log += d; });
      p.on('error', (e) => { log += '\n' + e.message; clearTimeout(timer); resolve(-1); });
      p.on('close', (c) => { clearTimeout(timer); resolve(c); });
    });
    const seconds = (Date.now() - t0) / 1000;
    log = log.split(dir).join('.');  // sem caminhos temporários no log
    if (code !== 0 || !fs.existsSync(output)) return send(res, 500, { ok: false, error: 'o cnx_cli terminou com código ' + code, log, seconds });
    send(res, 200, { ok: true, svg: fs.readFileSync(output, 'utf8'), log, seconds, args });
  } catch (e) {
    send(res, 500, { ok: false, error: e.message });
  } finally {
    busy = false;
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

const server = http.createServer((req, res) => {
  const host = (req.headers.host || '').split(':')[0];
  if (host !== '127.0.0.1' && host !== 'localhost') { res.writeHead(403); return res.end('host recusado'); }
  const origin = req.headers.origin;
  if (origin && !/^http:\/\/(127\.0\.0\.1|localhost)(:\d+)?$/.test(origin)) { res.writeHead(403); return res.end('origem recusada'); }
  const url = new URL(req.url, 'http://127.0.0.1');
  if (req.method === 'GET' && (url.pathname === '/' || url.pathname === '/index.html')) {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
    return res.end(fs.readFileSync(PAGE));
  }
  if (req.method === 'GET' && url.pathname === '/api/status') {
    return send(res, 200, { ok: true, exe: path.relative(ROOT, EXE), built: fs.statSync(EXE).mtime.toISOString(), busy });
  }
  if (req.method === 'POST' && url.pathname === '/api/vectorize') return void vectorize(req, res, url);
  res.writeHead(404); res.end('não encontrado');
});
server.listen(PORT, '127.0.0.1', () => {
  console.log('Bancada cnx em http://127.0.0.1:' + PORT);
  console.log('Motor: ' + EXE);
});
