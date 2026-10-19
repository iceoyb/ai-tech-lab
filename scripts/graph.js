// 知识图谱可视化
// 注：所有数据来自本地生成的 graph.json，无用户输入，安全可控

const CATEGORY_COLORS = {
  '架构': '#6ee7b7',
  '算法': '#60a5fa',
  '工具': '#fbbf24',
  '应用': '#f472b6',
  '趋势': '#c084fc',
  '工程': '#fb923c',
  '理论': '#34d399',
  '学术': '#a78bfa',
  '测试': '#f87171',
  '未分类': '#6b7280'
};

let allNodes = [], allEdges = [];
let simulation, svg, g, zoom;
let activeCategories = new Set();

document.addEventListener('DOMContentLoaded', () => {
  loadGraph();
});

async function loadGraph() {
  try {
    const res = await fetch('data/graph.json');
    const data = await res.json();
    
    allNodes = data.nodes;
    allEdges = data.edges;
    
    // 更新统计
    document.getElementById('gsNodes').textContent = data.stats.total_nodes.toLocaleString();
    document.getElementById('gsEdges').textContent = data.stats.total_edges.toLocaleString();
    
    // 构建分类筛选器
    const categories = Object.keys(data.stats.categories).sort((a, b) => data.stats.categories[b] - data.stats.categories[a]);
    const filterEl = document.getElementById('categoryFilter');
    categories.forEach(cat => {
      activeCategories.add(cat);
      const label = document.createElement('label');
      label.innerHTML = `
        <input type="checkbox" checked data-cat="${cat}">
        <span class="cat-dot" style="background: ${CATEGORY_COLORS[cat] || '#6b7280'}"></span>
        ${cat} (${data.stats.categories[cat]})
      `;
      label.querySelector('input').addEventListener('change', (e) => {
        if (e.target.checked) activeCategories.add(cat);
        else activeCategories.delete(cat);
        updateFilter();
      });
      filterEl.appendChild(label);
    });
    
    initGraph();
  } catch (e) {
    console.error('图谱加载失败', e);
    document.getElementById('gsNodes').textContent = '0';
    document.getElementById('gsEdges').textContent = '0';
  }
}

function initGraph() {
  const container = document.querySelector('.graph-container');
  const width = container.clientWidth;
  const height = container.clientHeight;
  
  svg = d3.select('#graphSvg')
    .attr('viewBox', [0, 0, width, height]);
  
  // defs for glow
  const defs = svg.append('defs');
  const filter = defs.append('filter').attr('id', 'glow');
  filter.append('feGaussianBlur').attr('stdDeviation', '2').attr('result', 'coloredBlur');
  const feMerge = filter.append('feMerge');
  feMerge.append('feMergeNode').attr('in', 'coloredBlur');
  feMerge.append('feMergeNode').attr('in', 'SourceGraphic');
  
  // 缩放
  zoom = d3.zoom()
    .scaleExtent([0.2, 5])
    .on('zoom', (e) => {
      g.attr('transform', e.transform);
    });
  
  svg.call(zoom);
  
  g = svg.append('g');
  
  // 初始化力导向图
  simulation = d3.forceSimulation(allNodes)
    .force('link', d3.forceLink(allEdges).id(d => d.id).distance(80).strength(0.3))
    .force('charge', d3.forceManyBody().strength(-120))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(18));
  
  renderGraph(allNodes, allEdges);
  
  // 响应式
  window.addEventListener('resize', () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    svg.attr('viewBox', [0, 0, w, h]);
    simulation.force('center', d3.forceCenter(w / 2, h / 2));
    simulation.alpha(0.3).restart();
  });
}

function renderGraph(nodes, edges) {
  g.selectAll('*').remove();
  
  // 连线
  const link = g.append('g')
    .selectAll('line')
    .data(edges)
    .join('line')
    .attr('stroke', '#2a3a55')
    .attr('stroke-opacity', 0.5)
    .attr('stroke-width', d => Math.sqrt(d.weight) || 1);
  
  // 节点
  const node = g.append('g')
    .selectAll('circle')
    .data(nodes)
    .join('circle')
    .attr('r', d => 6 + (d.mastery || 0) * 8)
    .attr('fill', d => CATEGORY_COLORS[d.category] || '#6b7280')
    .attr('fill-opacity', 0.85)
    .attr('stroke', d => CATEGORY_COLORS[d.category] || '#6b7280')
    .attr('stroke-width', 1.5)
    .style('cursor', 'pointer')
    .call(drag())
    .on('mouseover', showTooltip)
    .on('mouseout', hideTooltip)
    .on('click', (e, d) => {
      // 高亮关联节点
      highlightNode(d, node, link);
    });
  
  // 标签
  const label = g.append('g')
    .attr('class', 'labels')
    .selectAll('text')
    .data(nodes)
    .join('text')
    .text(d => truncate(d.title, 20))
    .attr('font-size', '10px')
    .attr('fill', '#94a3b8')
    .attr('text-anchor', 'middle')
    .attr('dy', -10)
    .style('pointer-events', 'none')
    .style('opacity', 0.8);
  
  // 位置更新
  simulation.on('tick', () => {
    link
      .attr('x1', d => d.source.x)
      .attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x)
      .attr('y2', d => d.target.y);
    
    node
      .attr('cx', d => d.x)
      .attr('cy', d => d.y);
    
    label
      .attr('x', d => d.x)
      .attr('y', d => d.y);
  });
}

function drag() {
  function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }
  function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
  }
  function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
  return d3.drag()
    .on('start', dragstarted)
    .on('drag', dragged)
    .on('end', dragended);
}

function showTooltip(event, d) {
  const tooltip = document.getElementById('tooltip');
  const container = document.querySelector('.graph-container');
  const rect = container.getBoundingClientRect();
  
  tooltip.innerHTML = `
    <h5>${d.title}</h5>
    <div style="color: var(--text-dim); margin-bottom: 6px;">分类：${d.category} · 深度：${'⭐'.repeat(d.depth || 1)}</div>
    <div>标签：${(d.tags || []).map(t => `<span style="display:inline-block;background:var(--bg-hover);padding:1px 6px;border-radius:4px;font-size:10px;margin:1px;">${t}</span>`).join('')}</div>
    <div class="tt-meta">添加于 ${(d.created_at || '').split('T')[0]}</div>
  `;
  tooltip.style.display = 'block';
  
  const x = event.clientX - rect.left + 15;
  const y = event.clientY - rect.top + 15;
  tooltip.style.left = Math.min(x, rect.width - 300) + 'px';
  tooltip.style.top = Math.min(y, rect.height - 200) + 'px';
}

function hideTooltip() {
  document.getElementById('tooltip').style.display = 'none';
}

function highlightNode(d, node, link) {
  // 简单实现：点击无操作，hover 已经够了
  // 可以扩展：点击后高亮关联节点
}

function updateFilter() {
  const filteredNodes = allNodes.filter(n => activeCategories.has(n.category));
  const nodeIds = new Set(filteredNodes.map(n => n.id));
  const filteredEdges = allEdges.filter(e => 
    nodeIds.has(typeof e.source === 'object' ? e.source.id : e.source) &&
    nodeIds.has(typeof e.target === 'object' ? e.target.id : e.target)
  );
  
  // 重置节点位置
  filteredNodes.forEach(n => {
    if (!n.x) { n.x = Math.random() * 800; n.y = Math.random() * 600; }
  });
  
  simulation.nodes(filteredNodes);
  simulation.force('link').links(filteredEdges);
  simulation.alpha(0.5).restart();
  
  renderGraph(filteredNodes, filteredEdges);
}

function resetZoom() {
  svg.transition().duration(500).call(
    zoom.transform,
    d3.zoomIdentity
  );
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('showLabels')?.addEventListener('change', (e) => {
    d3.select('.labels').style('opacity', e.target.checked ? 0.8 : 0);
  });
});

function truncate(str, n) {
  if (!str) return '';
  return str.length > n ? str.slice(0, n) + '...' : str;
}
