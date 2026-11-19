// Builds web/index.html and web/public/ from the worm repo, so each file has one source:
// the page from worm/static, the Python from web/py and worm/, the policy from results/.
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const web = path.dirname(path.dirname(new URL(import.meta.url).pathname));
const repo = path.dirname(web);
const pub = path.join(web, 'public');

fs.rmSync(pub, { recursive: true, force: true });
fs.mkdirSync(path.join(pub, 'py'), { recursive: true });
for (const f of fs.readdirSync(path.join(web, 'py')).filter((f) => f.endsWith('.py'))) {
  fs.copyFileSync(path.join(web, 'py', f), path.join(pub, 'py', f));
}
fs.copyFileSync(path.join(repo, 'results/policy_terrain/policy.json'), path.join(pub, 'py/policy.json'));
// The race ghosts: checkpoints of the run that made the final policy, and the flat v1 baseline.
fs.mkdirSync(path.join(pub, 'py/race'));
for (const step of fs.readdirSync(path.join(repo, 'results/race')).filter((d) => d.startsWith('step_'))) {
  fs.copyFileSync(path.join(repo, 'results/race', step, 'policy.json'), path.join(pub, 'py/race', `${step}.json`));
}
fs.copyFileSync(path.join(repo, 'results/policy/policy.json'), path.join(pub, 'py/race/flat.json'));
fs.cpSync(path.join(repo, 'worm/static/fonts'), path.join(pub, 'fonts'), { recursive: true });

const tmp = fs.mkdtempSync(path.join(web, '.wheel-'));
try {
  execSync(`uv build --wheel --out-dir ${tmp}`, { cwd: repo, stdio: 'pipe' });
  const wheel = fs.readdirSync(tmp).find((f) => f.endsWith('.whl'));
  fs.copyFileSync(path.join(tmp, wheel), path.join(pub, 'py/worm.whl'));
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}

// The entry script goes BEFORE the page's inline module script: document order decides which
// runs first, and window.wormApi must exist when the page reads it.
const page = fs.readFileSync(path.join(repo, 'worm/static/index.html'), 'utf8');
const marker = '<script type="module">';
if (!page.includes(marker)) throw new Error('worm/static/index.html has no inline module script');
fs.writeFileSync(path.join(web, 'index.html'), page.replace(marker, `<script type="module" src="./src/main.ts"></script>\n${marker}`));
console.log('prepared index.html and public/');
