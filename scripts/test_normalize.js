const fs = require('fs');
const path = 'last_diag.json';
if (!fs.existsSync(path)) { console.error('last_diag.json not found'); process.exit(1); }
let raw = fs.readFileSync(path,'utf8');
raw = raw.replace(/^\uFEFF/, '')
const resp = JSON.parse(raw);
const rawEigenInfo = resp?.spectralAnalysis?.eigenInformation;
if (!rawEigenInfo) { console.error('no spectral eigenInformation'); process.exit(1); }
const esList = rawEigenInfo.eigenspacePerEigenvalue || [];
const ebList = rawEigenInfo.eigenbasisPerEigenvalue || [];
const ev = rawEigenInfo.eigenvalues || [];
const alg = resp?.spectralAnalysis?.multiplicityAndPolynomials?.algebraicMultiplicity || null;
const geom = resp?.spectralAnalysis?.multiplicityAndPolynomials?.geometricMultiplicity || null;
const eigenvectors = rawEigenInfo.eigenvectors && rawEigenInfo.eigenvectors.data ? rawEigenInfo.eigenvectors.data : null;

function close(a,b, tol=1e-8){ if(!a||!b) return false; let s=0; for(let i=0;i<Math.min(a.length,b.length);i++){ s += Math.pow((a[i]||0)-(b[i]||0),2); } return Math.sqrt(s) <= tol }

const per = [];
console.error('eigenvalues:', JSON.stringify(ev));
console.error('eigenvectors (rows):', JSON.stringify(eigenvectors));
for (let i=0;i<esList.length;i++){
  const es = esList[i];
  const eb = (ebList && ebList[i]) ? ebList[i] : null;
  let eigen = null;
  if (ev && Array.isArray(ev) && ev.length === esList.length && ev[i]) eigen = ev[i];
  const vectors = es && es.vectors ? es.vectors : null;
  const basisVectors = eb && eb.vectors ? eb.vectors : (vectors && vectors.length ? vectors : null);
  const rep = (basisVectors && basisVectors.length) ? basisVectors[0] : (vectors && vectors.length ? vectors[0] : null);
  // determine matchedIndex
  let matchedIndex = null;
  if (ev && Array.isArray(ev) && ev.length === esList.length && ev[i]) matchedIndex = i;
  if (!eigen && rep && eigenvectors) {
    const cols = eigenvectors[0] ? eigenvectors[0].length : 0;
    for (let j=0;j<cols;j++){
      const col = eigenvectors.map(r => (r && r[j]!==undefined) ? r[j] : 0);
      const dist = Math.sqrt(col.reduce((s,v,k)=>s+Math.pow((v||0)-(rep[k]||0),2),0))
      console.error('compare col',j,col,'to rep',rep,'dist',dist)
      if (dist <= 1e-8) { eigen = ev && ev[j] ? ev[j] : null; matchedIndex = j; console.error('matched col',j,'->',eigen); break; }
    }
  }
  const algVal = (alg && matchedIndex !== null && alg[matchedIndex] !== undefined) ? alg[matchedIndex] : null
  const geoVal = (geom && matchedIndex !== null && geom[matchedIndex] !== undefined) ? geom[matchedIndex] : null
  per.push({ eigenvalue: eigen, representative: rep, eigenspace: vectors, eigenbasis: basisVectors, matchedIndex, algVal, geoVal });
}
console.log(JSON.stringify(per, null, 2));
