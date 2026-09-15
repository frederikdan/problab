/* ProbLab source explorer. No external libraries or network services. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const paths = {
    search:'m20 20-4-4 M17 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0',
    map:'M4 4h5v5H4z M15 4h5v5h-5z M10 15h5v5h-5z M6 9v3h12V9 M12 12v3',
    box:'m12 3 9 5v9l-9 5-9-5V8z M3 8l9 5 9-5 M12 13v9 M7 5l10 6',
    branch:'M6 6v12 M18 6v3a4 4 0 0 1-4 4h-4 M6 3a3 3 0 1 0 0 6 3 3 0 0 0 0-6 M6 15a3 3 0 1 0 0 6 3 3 0 0 0 0-6 M18 3a3 3 0 1 0 0 6 3 3 0 0 0 0-6',
    sun:'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8 M12 2v2 M12 20v2 M2 12h2 M20 12h2 M5 5l1 1 M18 18l1 1 M5 19l1-1 M18 6l1-1',
    panel:'M3 4h18v16H3z M15 4v16', help:'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18 M9 9a3 3 0 1 1 4 3c-1 0-1 1-1 2 M12 17h.01',
    refresh:'M20 8a8 8 0 0 0-14-3L3 8 M3 3v5h5 M4 16a8 8 0 0 0 14 3l3-3 M21 21v-5h-5',
    list:'M8 6h13 M8 12h13 M8 18h13 M3 6h.01 M3 12h.01 M3 18h.01',
    focus:'M8 3H3v5 M16 3h5v5 M21 16v5h-5 M8 21H3v-5 M9 9h6v6H9z',
  };
  const icon = name => `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="${paths[name] || paths.box}"/></svg>`;
  document.querySelectorAll('[data-icon]').forEach(el => el.innerHTML = icon(el.dataset.icon));
  const domains = {
    distributions: {color:'#d6bb92', label:'Distributions', description:'Define the possible outcomes.', long:'Probability distributions, their parameters, and sampling. Concrete distributions share the Distribution interface.'},
    random_variables: {color:'#b9d7bc', label:'Random variables', description:'Compose values into a shared graph.', long:'The main composition layer. RandomVariable builds expressions; nodes and realization contexts evaluate their shared dependencies.'},
    probability: {color:'#bcace0', label:'Probability', description:'Evaluate events and uncertainty.', long:'The P entry point, probability results, and the intervals used to describe uncertainty.'},
    functions: {color:'#a1c7d5', label:'Functions', description:'Apply familiar mathematical functions.', long:'Mathematical functions that work with scalar values and random-variable expressions.'},
    value_sets: {color:'#a6b8dc', label:'Value sets', description:'Describe and infer possible values.', long:'Numeric and object value sets describe possible values. Inference and membership helpers support mathematical operations and validation.'},
    _operations: {color:'#c5b2a1', label:'Operations', description:'The vocabulary of expressions.', long:'Operation definitions connect Python and NumPy functions with naming and value-set inference.'},
    validation: {color:'#adbfa0', label:'Validation', description:'Check the contracts at the boundaries.', long:'Reusable validators and the parameter-validation decorator check inputs across the library.'},
    statistics: {color:'#d0aaaa', label:'Statistics', description:'Quantify estimation uncertainty.', long:'Statistical routines for confidence intervals and quantiles.'},
    _events: {color:'#bbaec7', label:'Events', description:'Combine conditions into events.', long:'The internal event representation, with logical composition of boolean-valued nodes.'},
  };
  const symbols = {package:'▧', module:'{ }', class:'C', function:'ƒ', method:'m', property:'p', constant:'='};
  const relations = {imports:'Imports', inherits:'Inherits', calls:'Calls', references:'References'};
  let data = window.PROBLAB_GRAPH;
  if (!data) { $('page-description').textContent = 'Source map missing. Run build_graph.py, then reload this page.'; return; }
  let nodes, children, allEdges, visible = [], folded = [], positions = new Map();
  let state = {scope:data.root, selected:data.root, api:false, view:'graph', internal:true, tab:'overview', page:0, connections:false};
  let totalVisible=0;
  let enabled = new Set(Object.keys(relations)), transform = {x:0,y:0,scale:1}, graphSize = {width:1,height:1};
  let toastTimer, drag, dragged = false;

  function indexGraph() {
    nodes = new Map(data.nodes.map(node => [node.id,node]));
    children = new Map(data.nodes.map(node => [node.id,[]]));
    data.nodes.forEach(node => children.get(node.parent)?.push(node));
    allEdges = data.edges;
    // Memoized structural size includes nested definitions; imports are connections, not definitions.
    function count(id) {
      const node = nodes.get(id);
      node.descendants = (children.get(id) || []).reduce((n, child) => n + 1 + count(child.id), 0);
      return node.descendants;
    }
    count(data.root);
  }
  function domain(node) { return domains[node.id.split('.')[1]] || {color:'#b9d7bc',label:'ProbLab',description:'Probability, composed.',long:'A Python library for modelling, combining, and evaluating random variables.'}; }
  const isContainer = node => (children.get(node.id) || []).length > 0;
  const inScope = (id, scope) => id === scope || id.startsWith(scope + '.');
  function connections(node) { return allEdges.filter(e => (inScope(e.source,node.id) && !inScope(e.target,node.id)) || (inScope(e.target,node.id) && !inScope(e.source,node.id))); }
  function dependentCount(node) { return new Set(allEdges.filter(e => inScope(e.target,node.id) && !inScope(e.source,node.id)).map(e=>e.source)).size; }
  function description(node) {
    if (node.parent === data.root && domains[node.name]) return domain(node).description;
    if (node.doc) return node.doc.split('\n')[0];
    if (node.kind === 'class') return node.bases.length ? `Extends ${node.bases.join(', ')}` : `${(children.get(node.id)||[]).length} members defined here`;
    if (isContainer(node)) return `${node.descendants} definitions & containers`;
    if (node.kind === 'constant') return node.signature.split('=')[1]?.trim() || 'Module-level constant';
    return node.signature.replace(/^def \w+/, '').replace(/^async def \w+/, '') || 'Python source module';
  }
  function ordered(items) {
    return [...items].sort((a,b) => {
      if (state.scope === data.root && !state.api) return Object.keys(domains).indexOf(a.name)-Object.keys(domains).indexOf(b.name);
      return Number(b.public)-Number(a.public) || b.importance-a.importance || a.name.localeCompare(b.name);
    });
  }
  function getVisible() {
    let list = state.api ? data.nodes.filter(n=>n.public) : (children.get(state.scope) || []);
    if(state.connections) {
      const scopeNode=nodes.get(state.scope), related=connections(scopeNode).filter(e=>enabled.has(e.kind));
      const relatedIds=new Set(related.map(e=>inScope(e.source,state.scope)?e.target:e.source));
      // Ancestor containers are already represented by the selected definition.
      list=[...relatedIds].filter(id=>!inScope(state.scope,id)).map(id=>nodes.get(id));
    }
    list=ordered(list.filter(n => (state.internal || !n.internal) && !(n.kind === 'module' && n.descendants === 0 && !n.doc)));
    if(state.connections)list.unshift(nodes.get(state.scope));
    totalVisible=list.length;
    const pageSize=state.connections?8:9, pageable=state.connections?list.slice(1):list;
    state.page=Math.min(state.page,Math.max(0,Math.ceil(pageable.length/pageSize)-1));
    return state.view==='graph'?[...(state.connections?[nodes.get(state.scope)]:[]),...pageable.slice(state.page*pageSize,state.page*pageSize+pageSize)]:list;
  }
  function writeLocation() {
    const params = new URLSearchParams({scope:state.scope, selected:state.selected, view:state.view, page:state.page});
    if (state.api) params.set('api','1');
    if (state.connections) params.set('connections','1');
    if (!state.internal) params.set('internal','0');
    history.replaceState(null,'',`#${params}`);
  }
  function readLocation() {
    const p = new URLSearchParams(location.hash.slice(1));
    if (nodes.has(p.get('scope'))) state.scope = p.get('scope');
    if (nodes.has(p.get('selected'))) state.selected = p.get('selected');
    state.api = p.get('api') === '1';
    state.view = (p.get('view') || (window.matchMedia('(max-width: 760px)').matches?'list':'graph')) === 'list' ? 'list' : 'graph';
    state.internal = p.get('internal') !== '0';
    state.page=Math.max(0,Number.parseInt(p.get('page')||'0',10)||0);state.connections=p.get('connections')==='1';
  }
  function navigate(id, api=false) {
    if (!nodes.has(id)) return;
    state.scope = id; state.selected = id; state.api = api; state.tab='overview'; state.page=0;state.connections=false;
    render(); writeLocation();
  }
  function reveal(id) {
    const node=nodes.get(id); if (!node) return;
    state.scope=node.parent || data.root; state.selected=id; state.api=false; state.tab='overview'; state.page=0;state.connections=false;
    if (node.internal) state.internal=true;
    const candidates=ordered((children.get(state.scope)||[]).filter(n=>state.internal||!n.internal));
    state.page=Math.floor(Math.max(0,candidates.findIndex(n=>n.id===id))/9);
    render(); writeLocation();
  }
  function select(id) {
    state.selected=id; state.tab='overview';
    if ($('inspector').hidden) { $('inspector').hidden=false; requestAnimationFrame(fit); }
    renderInspector(); highlight(); writeLocation();
  }
  function toast(message) {
    clearTimeout(toastTimer); $('toast').textContent=message; $('toast').classList.add('visible');
    toastTimer=setTimeout(()=>$('toast').classList.remove('visible'),6500);
  }
  function renderNavigation() {
    $('api-count').textContent=data.nodes.filter(n=>n.public).length;
    $('module-count').textContent=data.counts.module || 0;
    $('summary-stats').innerHTML=[['class','classes'],['function','functions'],['method','methods']].map(([kind,label])=>`<div class="stat"><strong>${data.counts[kind]||0}</strong><span>${label}</span></div>`).join('');
    $('package-nav').innerHTML=[...(children.get(data.root)||[])].filter(n=>domains[n.name]).sort((a,b)=>Object.keys(domains).indexOf(a.name)-Object.keys(domains).indexOf(b.name)).map(n=>`<button class="nav-item ${inScope(state.scope,n.id)?'active':''}" data-nav="${esc(n.id)}" style="--domain:${domain(n).color}"><span class="domain-dot"></span>${esc(n.name)}<span class="counter">${n.descendants}</span></button>`).join('');
    $('architecture').classList.toggle('active',!state.api && state.scope===data.root);
    $('public-api').classList.toggle('active',state.api);
    $('snapshot-hash').textContent=data.sourceHash;
    $('source-version').textContent=`${Object.keys(data.modules).length} Python files · ${allEdges.length} connections`;
    $('analysis-note').textContent=data.limitations;
    $('snapshot-status').textContent=location.protocol==='file:'?'Source snapshot':'Local source · refreshable';
  }
  function renderBreadcrumbs() {
    const chain=[]; let id=state.scope;
    while(nodes.has(id)) { chain.unshift(nodes.get(id)); id=nodes.get(id).parent; }
    $('breadcrumbs').innerHTML=chain.map(n=>`<button data-nav="${esc(n.id)}">${esc(n.name)}</button>`).join('<span class="divider">/</span>')+(state.api?'<span class="divider">/</span><span>Public API</span>':state.connections?'<span class="divider">/</span><span>Connections</span>':'');
    $('page-title').innerHTML=state.api?'The public API<span>.</span>':state.scope===data.root?'A map of ProbLab<span>.</span>':`${esc(nodes.get(state.scope).name)}<span>.</span>`;
    $('page-description').textContent=state.api?'The entry points exported directly from problab.':state.scope===data.root?'Start with the big picture. Follow a connection. Open up the details.':description(nodes.get(state.scope));
    $('scope-label').textContent=state.connections?'CONNECTIONS':state.api?'PUBLIC ENTRY POINTS':state.scope===data.root?'PACKAGE OVERVIEW':`${nodes.get(state.scope).kind.toUpperCase()} CONTENTS`;
    const pageCount=Math.ceil((totalVisible-(state.connections?1:0))/(state.connections?8:9));
    $('visible-count').innerHTML=state.view==='graph'&&totalVisible>9?`<button id="previous-page" aria-label="Previous components" ${state.page===0?'disabled':''}>←</button> ${state.page+1} / ${pageCount} · ${totalVisible} components <button id="next-page" aria-label="Next components" ${state.page+1>=pageCount?'disabled':''}>→</button>`:`${visible.length} components`;
    if($('previous-page'))$('previous-page').onclick=()=>{state.page--;render();writeLocation();};
    if($('next-page'))$('next-page').onclick=()=>{state.page++;render();writeLocation();};
    $('internal-toggle').checked=state.internal;
  }
  function foldEdges() {
    const ids=new Set(visible.map(n=>n.id));
    const owner=id=>{ while(id) { if(ids.has(id))return id; id=nodes.get(id)?.parent; } return null; };
    const groups=new Map();
    allEdges.forEach(e=>{
      if(!enabled.has(e.kind))return;
      const a=owner(e.source), b=owner(e.target);
      if(!a||!b||a===b)return;
      const key=`${a}|${b}`;
      if(!groups.has(key)) groups.set(key,{source:a,target:b,count:0,kinds:new Set()});
      const group=groups.get(key); group.count++; group.kinds.add(e.kind);
    });
    return [...groups.values()];
  }
  function renderMap() {
    folded=foldEdges(); positions=new Map();
    const compact=visible.length>6;
    const columns=visible.length<=2?visible.length:3, cardW=224, cardH=compact?86:126, gapX=42, gapY=compact?26:45;
    visible.forEach((n,i)=>positions.set(n.id,{x:18+(i%columns)*(cardW+gapX),y:12+Math.floor(i/columns)*(cardH+gapY)}));
    graphSize={width:Math.max(1,columns*(cardW+gapX)-gapX+36),height:Math.max(1,Math.ceil(visible.length/Math.max(1,columns))*(cardH+gapY)-gapY+24)};
    $('cards').classList.toggle('compact',compact);
    $('graph-stage').style.width=graphSize.width+'px'; $('graph-stage').style.height=graphSize.height+'px';
    $('edges').setAttribute('width',graphSize.width); $('edges').setAttribute('height',graphSize.height);
    const maxDependents=Math.max(1,...visible.map(dependentCount));
    $('cards').innerHTML=visible.map((n,i)=>{
      const pos=positions.get(n.id), count=dependentCount(n), bars=count?Math.max(1,Math.round(count/maxDependents*5)):0;
      const hub=state.scope===data.root&&!state.api&&count>=maxDependents*.65;
      return `<article class="node-card" tabindex="0" role="button"
          aria-label="Inspect ${esc(n.name)}, ${n.kind}" data-select="${esc(n.id)}"
          style="left:${pos.x}px;top:${pos.y}px;--domain:${domain(n).color};animation-delay:${Math.min(i,12)*22}ms">
        <div class="node-top"><span class="symbol-icon">${symbols[n.kind]}</span>${n.kind}
          ${n.public?'<span class="api-badge">API</span>':hub?'<span class="api-badge">HUB</span>':''}
        </div>
        <h2 class="node-name" title="${esc(n.name)}">${esc(n.name)}</h2>
        <div class="node-description">${esc(description(n))}</div>
        <div class="node-bottom">
          <span class="importance" title="${count} dependent definitions">${Array.from({length:5},(_,j)=>`<i class="${j<bars?'on':''}"></i>`).join('')}</span>
          <span>${count} dependents</span>
          ${isContainer(n)?`<button class="explore-button" data-nav="${esc(n.id)}" aria-label="Explore ${esc(n.name)}">Explore ↗</button>`:'<span class="explore-button">Inspect ↗</span>'}
        </div>
      </article>`;
    }).join('');
    $('edges').innerHTML=`<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="context-stroke"/></marker></defs>`+folded.map(e=>{
      const a=positions.get(e.source),b=positions.get(e.target); let x1,y1,x2,y2,c1,c2;
      if(a.x!==b.x) {
        const direction=b.x>a.x?1:-1;
        x1=a.x+(direction>0?cardW:0); y1=a.y+cardH/2;
        x2=b.x+(direction>0?0:cardW); y2=b.y+cardH/2;
        const bend=Math.max(33,Math.abs(x2-x1)*.45);
        c1=`${x1+direction*bend},${y1}`; c2=`${x2-direction*bend},${y2}`;
      } else {
        const direction=b.y>a.y?1:-1;
        x1=a.x+cardW/2; y1=a.y+(direction>0?cardH:0); x2=b.x+cardW/2; y2=b.y+(direction>0?0:cardH);
        c1=`${x1},${y1+direction*32}`;c2=`${x2},${y2-direction*32}`;
      }
      return `<path class="graph-edge ${e.kinds.has('inherits')?'inherits':''}" data-from="${esc(e.source)}" data-to="${esc(e.target)}" d="M${x1},${y1}C${c1} ${c2} ${x2},${y2}" marker-end="url(#arrow)"><title>${esc(nodes.get(e.source).name)} → ${esc(nodes.get(e.target).name)}: ${e.count} connections</title></path>`;
    }).join('');
    $('empty-state').hidden=visible.length>0;
    $('empty-state').innerHTML='<strong>No components at this level.</strong><p>Enable Internals or use the breadcrumbs to move up.</p>';
    $('list-canvas').innerHTML=visible.map(n=>`<div class="definition-row" role="button" tabindex="0" data-select="${esc(n.id)}" style="--domain:${domain(n).color}"><span class="symbol-icon">${symbols[n.kind]}</span><div class="definition-row-main"><strong>${esc(n.name)}</strong><small>${esc(description(n))}</small></div><span class="badge">${n.kind}</span>${n.public?'<span class="badge public">API</span>':''}${isContainer(n)?`<button class="explore-button" data-nav="${esc(n.id)}">Explore ↗</button>`:''}</div>`).join('')||'<p class="small-note">No visible definitions. Try enabling Internals.</p>';
    $('graph-canvas').hidden=state.view!=='graph'; $('list-canvas').hidden=state.view!=='list';
    document.querySelector('.zoom-controls').hidden=state.view!=='graph';
    document.querySelector('.edge-legend').hidden=state.view!=='graph';
    $('canvas-hint').hidden=state.view!=='graph';
    $('graph-view').classList.toggle('active',state.view==='graph'); $('list-view').classList.toggle('active',state.view==='list');
    highlight(); requestAnimationFrame(fit);
  }
  function highlight(hoverId=null) {
    const id=hoverId||state.selected, active=positions.has(id), connected=new Set([id]);
    folded.forEach(e=>{if(e.source===id||e.target===id){connected.add(e.source);connected.add(e.target);}});
    document.querySelectorAll('.node-card,.definition-row').forEach(card=>{
      card.classList.toggle('selected',card.dataset.select===state.selected);
      card.classList.toggle('dim',active&&!connected.has(card.dataset.select));
    });
    document.querySelectorAll('.graph-edge').forEach(edge=>{
      const relevant=edge.dataset.from===id||edge.dataset.to===id;
      edge.classList.toggle('highlight',active&&relevant); edge.classList.toggle('dim',active&&!relevant);
    });
  }
  function row(node, right='') {
    return `<button class="link-row" data-reveal="${esc(node.id)}" style="--domain:${domain(node).color}" title="${esc(node.id)}"><span class="symbol-icon">${symbols[node.kind]}</span><span class="link-row-name">${esc(node.name)}<small>${esc(node.parent?.split('.').slice(-2).join('.')||'')}</small></span><span class="link-row-kind">${esc(right||node.kind)} ↗</span></button>`;
  }
  function renderInspector() {
    const node=nodes.get(state.selected)||nodes.get(data.root), d=domain(node);
    if(node.id===data.root && !state.api) {
      const entry=name=>data.nodes.find(n=>n.public&&n.name===name);
      $('inspector-content').innerHTML=`<div class="intro-mark">⌘</div><h2 class="intro-title">One library.<br>Connected ideas.</h2><p class="description">A source-derived guide to how ProbLab fits together. Choose a component to see what it contains and what it depends on.</p><div class="metrics"><div class="metric"><strong>${data.counts.package||0}</strong><span>packages</span></div><div class="metric"><strong>${data.nodes.filter(n=>n.public).length}</strong><span>public entry points</span></div></div><div class="inspector-section"><h3>GOOD PLACES TO START</h3>${[['RandomVariable','Build and combine expressions'],['Distribution','Define how values are sampled'],['P','Ask a probability question']].map(([name,desc])=>entry(name)?`<button class="intro-route" data-reveal="${esc(entry(name).id)}"><span>${name}<small>${desc}</small></span><span>↗</span></button>`:'').join('')}</div><p class="small-note">Brighter dependency bars mark components used by more definitions. API badges identify root-level public exports.</p>`;
      return;
    }
    const edges=connections(node).filter(e=>enabled.has(e.kind));
    const outgoing=[...new Set(edges.filter(e=>inScope(e.source,node.id)).map(e=>e.target))].map(id=>nodes.get(id));
    const incoming=[...new Set(edges.filter(e=>inScope(e.target,node.id)).map(e=>e.source))].map(id=>nodes.get(id));
    const members=ordered(children.get(node.id)||[]).filter(n=>state.internal||!n.internal);
    const docs=node.doc||(node.parent===data.root?d.long:description(node));
    $('inspector-content').innerHTML=`<div class="inspector-symbol" style="--domain:${d.color}">${symbols[node.kind]}</div><h2 class="inspector-title">${esc(node.name)}</h2><div class="inspector-path">${esc(node.path)}${node.kind==='package'?'':`:${node.line}`}</div><div class="badges"><span class="badge">${node.kind}</span>${node.public?'<span class="badge public">Public API</span>':node.internal?'<span class="badge">Internal</span>':''}</div><div class="inspector-tabs"><button data-tab="overview" class="${state.tab==='overview'?'active':''}">Overview</button><button data-tab="source" class="${state.tab==='source'?'active':''}">Source</button></div>${state.tab==='source'?sourceMarkup(node):`<p class="description">${esc(docs)}</p>${node.signature?`<div class="signature">${esc(node.signature)}</div>`:''}${isContainer(node)?`<button class="primary-button" data-nav="${esc(node.id)}">Explore ${members.length} components <span>↗</span></button>`:''}<button class="connection-button" data-connect="${esc(node.id)}">Map connections <span>↗</span></button><div class="metrics"><div class="metric"><strong>${outgoing.length}</strong><span>dependencies</span></div><div class="metric"><strong>${incoming.length}</strong><span>dependents</span></div></div>${node.fields.length?`<section class="inspector-section"><h3>Declared fields</h3><div class="field">${node.fields.map(esc).join('\n')}</div></section>`:''}${members.length?`<section class="inspector-section"><h3>Defined here <span>${members.length}</span></h3>${members.map(n=>row(n)).join('')}</section>`:''}${outgoing.length?`<section class="inspector-section"><h3>Depends on <span>${outgoing.length}</span></h3>${outgoing.sort((a,b)=>b.importance-a.importance).map(n=>row(n)).join('')}</section>`:''}${incoming.length?`<section class="inspector-section"><h3>Used by <span>${incoming.length}</span></h3>${incoming.sort((a,b)=>b.importance-a.importance).map(n=>row(n)).join('')}</section>`:''}${node.unresolvedCalls?`<p class="small-note">${node.unresolvedCalls} external or dynamic call sites are not linked by the static analyzer.</p>`:''}<p class="small-note">Connection counts follow the filters below.</p>`}`;
  }
  function sourceMarkup(node) {
    const lines=data.modules[node.module].source.split('\n').slice(node.line-1,node.endLine);
    const colorize=line=>line.split(/(#.*$|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b(?:class|def|return|from|import|if|else|elif|for|in|is|not|and|or|with|try|except|raise|pass|None|True|False|self|super|yield|lambda|async|await)\b)/g).map(part=>{
      if(!part)return '';
      const cls=part.startsWith('#')?'syntax-comment':/^["']/.test(part)?'syntax-string':/^(class|def|return|from|import|if|else|elif|for|in|is|not|and|or|with|try|except|raise|pass|None|True|False|self|super|yield|lambda|async|await)$/.test(part)?'syntax-keyword':'';
      return cls?`<span class="${cls}">${esc(part)}</span>`:esc(part);
    }).join('');
    return `<p class="small-note">Source snapshot · lines ${node.line}–${node.endLine}</p><div class="source-code">${lines.map((line,i)=>`<div class="source-line"><span class="line-number">${node.line+i}</span><span class="source-text">${colorize(line)}</span></div>`).join('')}</div>`;
  }
  function renderFilters() {
    $('relation-filters').innerHTML=Object.entries(relations).map(([key,label])=>`<button class="relation-filter ${enabled.has(key)?'active':''}" aria-pressed="${enabled.has(key)}" data-relation="${key}">${label}</button>`).join('');
  }
  function render() { visible=getVisible(); renderNavigation(); renderBreadcrumbs(); renderMap(); renderInspector(); renderFilters(); }
  function applyTransform(animate=false) {
    $('graph-stage').classList.toggle('fit-animation',animate);
    $('graph-stage').style.transform=`translate(${transform.x}px,${transform.y}px) scale(${transform.scale})`;
  }
  function fit() {
    if(state.view!=='graph')return;
    const box=$('graph-canvas').getBoundingClientRect(); if(!box.width||!box.height)return;
    const scale=Math.min(1.15,(box.width-22)/graphSize.width,(box.height-16)/graphSize.height);
    transform={scale:Math.max(.12,scale),x:(box.width-graphSize.width*scale)/2,y:(box.height-graphSize.height*scale)/2};
    applyTransform(true);
  }
  function zoom(multiplier, x, y) {
    const box=$('graph-canvas').getBoundingClientRect(); x??=box.width/2; y??=box.height/2;
    const scale=Math.min(2.6,Math.max(.12,transform.scale*multiplier)), ratio=scale/transform.scale;
    transform={scale,x:x-(x-transform.x)*ratio,y:y-(y-transform.y)*ratio}; applyTransform();
  }
  function search() {
    const q=$('search-input').value.toLowerCase().trim();
    let list=data.nodes.filter(n=>q?n.id.toLowerCase().includes(q):n.public);
    list.sort((a,b)=>Number(b.name.toLowerCase()===q)-Number(a.name.toLowerCase()===q)||b.importance-a.importance);
    $('search-results').innerHTML=list.slice(0,60).map(n=>`<button class="search-result" data-reveal="${esc(n.id)}" style="--domain:${domain(n).color}"><span class="symbol-icon">${symbols[n.kind]}</span><span><strong>${esc(n.name)}</strong><small>${esc(n.id)}</small></span><span class="badge">${n.kind}</span></button>`).join('')||'<p class="small-note">No definitions match that search.</p>';
  }
  function openSearch() { $('search-dialog').showModal(); $('search-input').value=''; search(); $('search-input').focus(); }
  document.addEventListener('click',event=>{
    if(dragged) {dragged=false;return;}
    const button=event.target.closest('[data-nav],[data-select],[data-reveal],[data-tab],[data-relation],[data-connect]'); if(!button)return;
    if(button.dataset.nav) navigate(button.dataset.nav);
    else if(button.dataset.reveal) { reveal(button.dataset.reveal); $('search-dialog').close(); }
    else if(button.dataset.select) select(button.dataset.select);
    else if(button.dataset.connect) {state.scope=button.dataset.connect;state.selected=state.scope;state.connections=true;state.api=false;state.page=0;render();writeLocation();}
    else if(button.dataset.tab) {state.tab=button.dataset.tab;renderInspector();}
    else if(button.dataset.relation) {const key=button.dataset.relation;enabled.has(key)?enabled.delete(key):enabled.add(key);render();}
  });
  document.addEventListener('dblclick',event=>{const card=event.target.closest('[data-select]');if(card && isContainer(nodes.get(card.dataset.select)))navigate(card.dataset.select);});
  document.addEventListener('keydown',event=>{
    if(event.key==='Enter' && event.target.matches('[data-select]'))select(event.target.dataset.select);
    if(event.target.matches('input,textarea')||document.querySelector('dialog[open]'))return;
    if(event.key==='/'||((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='k')){event.preventDefault();openSearch();}
    if(event.key.toLowerCase()==='f')fit();
  });
  $('cards').addEventListener('pointerover',event=>{const card=event.target.closest('[data-select]');if(card)highlight(card.dataset.select);});
  $('cards').addEventListener('pointerleave',()=>highlight());
  $('graph-canvas').addEventListener('pointerdown',event=>{
    if(event.button!==0||event.target.closest('button'))return;
    drag={x:event.clientX,y:event.clientY,tx:transform.x,ty:transform.y}; dragged=false;
  });
  window.addEventListener('pointermove',event=>{
    if(!drag)return;
    const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
    if(Math.abs(dx)+Math.abs(dy)>5){dragged=true;$('graph-canvas').classList.add('panning');transform.x=drag.tx+dx;transform.y=drag.ty+dy;applyTransform();}
  });
  window.addEventListener('pointerup',()=>{drag=null;$('graph-canvas').classList.remove('panning');setTimeout(()=>dragged=false,0);});
  window.addEventListener('pointercancel',()=>{drag=null;dragged=false;});
  $('graph-canvas').addEventListener('wheel',event=>{event.preventDefault();const box=$('graph-canvas').getBoundingClientRect();zoom(Math.exp(-event.deltaY*.0015),event.clientX-box.left,event.clientY-box.top);},{passive:false});
  $('zoom-in').onclick=()=>zoom(1.2); $('zoom-out').onclick=()=>zoom(1/1.2); $('fit-view').onclick=fit;
  $('home-link').onclick=event=>{event.preventDefault();navigate(data.root);}; $('architecture').onclick=()=>navigate(data.root);
  $('public-api').onclick=()=>navigate(data.root,true);
  $('internal-toggle').onchange=()=>{state.internal=$('internal-toggle').checked;render();writeLocation();};
  $('graph-view').onclick=()=>{state.view='graph';render();writeLocation();}; $('list-view').onclick=()=>{state.view='list';render();writeLocation();};
  $('inspector-toggle').onclick=()=>{$('inspector').hidden=!$('inspector').hidden;requestAnimationFrame(fit);};
  $('search-open').onclick=openSearch; $('search-close').onclick=()=>$('search-dialog').close(); $('search-input').oninput=search;
  $('search-input').onkeydown=event=>{if(event.key==='ArrowDown'){event.preventDefault();$('search-results').querySelector('button')?.focus();}if(event.key==='Enter')$('search-results').querySelector('button')?.click();};
  $('help-open').onclick=$('about-map').onclick=()=>$('help-dialog').showModal(); $('help-close').onclick=()=>$('help-dialog').close();
  for(const id of ['search-dialog','help-dialog'])$(id).addEventListener('click',event=>{if(event.target===$(id)){const rect=$(id).getBoundingClientRect();if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)$(id).close();}});
  try {document.documentElement.dataset.theme=localStorage.getItem('problab-explorer-theme')||'dark';}catch{/* File storage can be disabled. */}
  $('theme-toggle').onclick=()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=theme;try{localStorage.setItem('problab-explorer-theme',theme);}catch{}};
  $('refresh').onclick=async()=>{
    if(location.protocol==='file:'){toast('For live refresh, run: python docs/library_explorer/serve.py. Or regenerate graph-data.js with build_graph.py.');return;}
    $('refresh').disabled=true;$('refresh').classList.add('spin');
    try {
      const response=await fetch('./api/graph',{cache:'no-store'});
      if(!response.ok)throw new Error('Refresh endpoint unavailable. Run the included serve.py for live refresh.');
      const next=await response.json();if(!Array.isArray(next.nodes))throw new Error('The server did not return a source map.');
      data=next;indexGraph(); if(!nodes.has(state.scope))state.scope=data.root;if(!nodes.has(state.selected))state.selected=state.scope;
      render();writeLocation();toast('Source map refreshed from the current Python files.');
    } catch(error){toast(error.message);} finally{$('refresh').disabled=false;$('refresh').classList.remove('spin');}
  };
  new ResizeObserver(()=>requestAnimationFrame(fit)).observe($('graph-canvas'));
  indexGraph();readLocation();render();
  window.addEventListener('hashchange',()=>{readLocation();render();});
})();
