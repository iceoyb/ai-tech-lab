(function (global) {
  var PLACEHOLDER = /is being prepared|正在准备|OverviewPaper review|BackgroundBest practice content for|auto-generated/i;

  function blob(item) {
    if (!item) return '';
    return [item.summary, item.description, item.title].filter(Boolean).join('\n');
  }

  function isPlaceholder(item) {
    return PLACEHOLDER.test(blob(item));
  }

  function isStubScript(item) {
    if (!item || isPlaceholder(item)) return true;
    var desc = item.description || '';
    var m = desc.match(/\((\d+)\s*lines\)/i);
    if (m && Number(m[1]) < 80) return true;
    return false;
  }

  function visiblePapers(papers) {
    var claimed = {};
    (papers || []).forEach(function (p) {
      if (!isPlaceholder(p) && p.deep_url) claimed[p.deep_url] = true;
    });
    var seen = {};
    var out = [];
    (papers || []).forEach(function (p) {
      if (isPlaceholder(p)) return;
      if (!p.deep_url && p.url && claimed[p.url]) return;
      var href = p.deep_url || p.url;
      if (!href || seen[href]) return;
      seen[href] = true;
      out.push(p);
    });
    return out;
  }

  function visiblePractices(practices) {
    return (practices || []).filter(function (p) { return !isPlaceholder(p); });
  }

  function visibleWeekly(reports) {
    return (reports || []).filter(function (r) { return !isPlaceholder(r); });
  }

  function featuredWeekly(reports) {
    return visibleWeekly(reports).filter(function (r) {
      return (r.week && r.week <= 35) || /存档/.test(r.summary || '');
    });
  }

  function graphCounts(data) {
    var nodes = data && Array.isArray(data.nodes) ? data.nodes.length : 0;
    var edges = data && Array.isArray(data.edges) ? data.edges.length : 0;
    if (!nodes && data && data.stats) nodes = data.stats.total_nodes || 0;
    if (!edges && data && data.stats) edges = data.stats.total_edges || 0;
    return { nodes: nodes, edges: edges };
  }

  function categoryCounts(nodes) {
    var counts = {};
    (nodes || []).forEach(function (n) {
      var cat = n.category || '未分类';
      counts[cat] = (counts[cat] || 0) + 1;
    });
    return counts;
  }

  function mergeScripts(featured, published) {
    var seen = {};
    var out = [];
    (featured || []).concat(published || []).forEach(function (s) {
      if (isStubScript(s)) return;
      var key = s.url || s.name;
      if (!key || seen[key]) return;
      seen[key] = true;
      out.push(s);
    });
    return out;
  }

  global.ContentFilter = {
    isPlaceholder: isPlaceholder,
    isStubScript: isStubScript,
    visiblePapers: visiblePapers,
    visiblePractices: visiblePractices,
    visibleWeekly: visibleWeekly,
    featuredWeekly: featuredWeekly,
    graphCounts: graphCounts,
    categoryCounts: categoryCounts,
    mergeScripts: mergeScripts
  };
})(window);
