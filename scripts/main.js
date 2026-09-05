// 首页逻辑
document.addEventListener('DOMContentLoaded', () => {
  loadStats();
  loadUpdates();
});

async function loadStats() {
  try {
    const res = await fetch('data/graph.json');
    const data = await res.json();
    document.getElementById('statNodes').textContent = data.stats.total_nodes.toLocaleString();
    document.getElementById('statEdges').textContent = data.stats.total_edges.toLocaleString();
    
    // 工具数和洞察数从 index 数据读
    const idxRes = await fetch('data/index.json');
    const idx = await idxRes.json();
    document.getElementById('statTools').textContent = idx.tools_count || 0;
    document.getElementById('statInsights').textContent = idx.insights_count || 0;
  } catch (e) {
    console.log('数据加载失败', e);
  }
}

async function loadUpdates() {
  const container = document.getElementById('latestUpdates');
  try {
    const res = await fetch('data/index.json');
    const data = await res.json();
    
    if (!data.updates || data.updates.length === 0) {
      container.innerHTML = '<p style="color: var(--text-dim); grid-column: 1/-1; text-align: center; padding: 40px;">正在积累内容，敬请期待...</p>';
      return;
    }
    
    container.innerHTML = data.updates.map(u => `
      <div class="update-card" onclick="location.href='${u.url}'">
        <span class="update-tag tag-${u.type}">${typeLabel(u.type)}</span>
        <h4>${u.title}</h4>
        <p>${u.summary || ''}</p>
        <div class="update-meta">
          <span>${u.date}</span>
          <span>${u.category || ''}</span>
        </div>
      </div>
    `).join('');
  } catch (e) {
    console.log(e);
  }
}

function typeLabel(t) {
  const map = { tool: '工具', paper: '论文', practice: '实践', insight: '洞察' };
  return map[t] || t;
}
