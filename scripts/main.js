// 首页逻辑
document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadUpdates();
});

async function loadStats() {
  try {
    const res = await fetch('data/graph.json');
    const data = await res.json();
    const counts = ContentFilter.graphCounts(data);
    document.getElementById('statNodes').textContent = counts.nodes.toLocaleString();
    document.getElementById('statEdges').textContent = counts.edges.toLocaleString();
    
    // 工具数和洞察数从 index 数据读
    const idxRes = await fetch('data/index.json');
    const idx = await idxRes.json();
    document.getElementById('statTools').textContent = idx.tools_count || 0;
    document.getElementById('statInsights').textContent = idx.insights_count || 0;
    // 四大板块卡片里的节点数同步（避免硬编码过期）
    const sn = document.getElementById('sectionNodes');
    if (sn) sn.textContent = counts.nodes.toLocaleString();
  } catch (e) {
    console.log('数据加载失败', e);
  }
}

function clip(text, n) {
  const s = (text || '').replace(/\s+/g, ' ').trim();
  return s.length > n ? s.slice(0, n) + '…' : s;
}

async function loadUpdates() {
  const container = document.getElementById('latestUpdates');
  try {
    const [papers, practices, weekly] = await Promise.all([
      fetch('data/papers.json').then(r => r.json()),
      fetch('data/practices.json').then(r => r.json()),
      fetch('data/weekly.json').then(r => r.json())
    ]);
    const items = []
      .concat(ContentFilter.visiblePapers(papers.papers).map(p => ({
        type: 'paper',
        title: p.title,
        summary: clip(p.summary, 140),
        date: p.date,
        category: (p.tags || []).slice(0, 2).join(' / '),
        url: 'papers/' + (p.deep_url || p.url)
      })))
      .concat(ContentFilter.visiblePractices(practices.practices).map(p => ({
        type: 'practice',
        title: p.title,
        summary: clip(p.summary, 140),
        date: p.date,
        category: (p.tags || []).slice(0, 2).join(' / '),
        url: 'practices/' + p.url
      })))
      .concat(ContentFilter.featuredWeekly(weekly.reports).map(r => ({
        type: 'weekly',
        title: '第 ' + r.week + ' 周进化周报',
        summary: r.summary || '',
        date: r.date,
        category: r.period || '',
        url: 'weekly/' + r.url
      })));

    if (!items.length) {
      container.innerHTML = '<p style="color: var(--text-dim); grid-column: 1/-1; text-align: center; padding: 40px;">正在积累内容，敬请期待...</p>';
      return;
    }

    container.innerHTML = items.map(u => `
      <div class="update-card" onclick="location.href='${u.url}'">
        <span class="update-tag tag-${u.type}">${typeLabel(u.type)}</span>
        <h4>${u.title}</h4>
        <p>${u.summary || ''}</p>
        <div class="update-meta">
          <span>${u.date || ''}</span>
          <span>${u.category || ''}</span>
        </div>
      </div>
    `).join('');
  } catch (e) {
    console.log(e);
  }
}

function typeLabel(t) {
  const map = { tool: '工具', paper: '论文', practice: '实践', insight: '洞察', weekly: '周报' };
  return map[t] || t;
}
