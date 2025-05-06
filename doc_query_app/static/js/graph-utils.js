// Color mapping for node groups
const GROUP_COLORS = {
  1: '#4285F4', // Web - Blue
  2: '#EA4335', // Mobile - Red
  3: '#FBBC05', // Cybersecurity - Yellow
  4: '#34A853', // IoT - Green
};

// Field names for groups
const GROUP_FIELDS = {
  1: 'Web Application',
  2: 'Mobile Application',
  3: 'Cybersecurity',
  4: 'IoT',
};

// Get node color based on group
function getNodeColor(group) {
  return GROUP_COLORS[group] || '#ffffff';
}

// Get field name from group
function getFieldFromGroup(group) {
  return GROUP_FIELDS[group] || 'Unknown';
}

// Handle node click event
function handleNodeClick(
  node,
  graph,
  infoPanel,
  projectTitle,
  projectField,
  projectDescription
) {
  // Display node info
  projectTitle.textContent = node.name;
  projectField.textContent = getFieldFromGroup(node.group);
  projectDescription.textContent = node.description;
  infoPanel.style.display = 'block';

  // Focus camera on node
  const distance = 120;
  const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
  graph.cameraPosition(
    { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
    node,
    1000
  );
}

// Reset camera position
function resetCamera(graph, infoPanel) {
  graph.cameraPosition({ x: 0, y: 0, z: 300 }, { x: 0, y: 0, z: 0 }, 1000);
  infoPanel.style.display = 'none';
}

// Animation function for rotation
function spin(graph, rotating) {
  if (!rotating) return;

  const curRotation = graph.rotation();
  graph.rotation({
    x: curRotation.x,
    y: curRotation.y + 0.001,
    z: curRotation.z,
  });

  requestAnimationFrame(() => spin(graph, rotating));
}

// Load graph data from API
function loadGraphData(graph, loadingDiv, infoPanel) {
  loadingDiv.style.display = 'block';

  fetch('/graph')
    .then((res) => {
      if (!res.ok) {
        throw new Error(`HTTP error! Status: ${res.status}`);
      }
      return res.json();
    })
    .then((data) => {
      graph.graphData(data);
      loadingDiv.style.display = 'none';

      // Initial positioning
      setTimeout(() => {
        graph.zoomToFit(400, 100);
      }, 1000);
    })
    .catch((error) => {
      console.error('Error loading graph data:', error);
      loadingDiv.innerHTML =
        'Error loading graph data. Please ensure the API server is running.';
    });
}
