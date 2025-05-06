import React, { useEffect, useRef, useState } from 'react';
import ForceGraph3D from '3d-force-graph';
import * as THREE from 'three';

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

const GraphView = () => {
  const containerRef = useRef(null);
  const graphRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [rotating, setRotating] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let graph = null;
    let animationFrameId = null;

    const initGraph = async () => {
      try {
        if (!containerRef.current) return;

        // Initialize the 3D force graph
        graph = ForceGraph3D({ controlType: 'orbit' })(containerRef.current)
          .backgroundColor('#000000')
          .nodeColor((node) => GROUP_COLORS[node.group] || '#ffffff')
          .nodeLabel((node) => node.name)
          .nodeRelSize(6)
          .nodeOpacity(0.9)
          .linkWidth((link) => link.value * 2)
          .linkOpacity(0.3)
          .linkDirectionalParticles(2)
          .linkDirectionalParticleWidth((link) => link.value * 2)
          .onNodeClick((node) => {
            setSelectedNode(node);
            // Focus camera on node
            const distance = 120;
            const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
            graph.cameraPosition(
              {
                x: node.x * distRatio,
                y: node.y * distRatio,
                z: node.z * distRatio,
              },
              node,
              1000
            );
          })
          .onBackgroundClick(() => setSelectedNode(null));

        graphRef.current = graph;

        // Load graph data
        const response = await fetch('http://localhost:8000/graph');
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();

        graph.graphData(data);
        setLoading(false);

        // Initial positioning
        setTimeout(() => {
          if (graph) {
            graph.zoomToFit(400, 100);
          }
        }, 1000);
      } catch (err) {
        console.error('Error initializing graph:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    initGraph();

    return () => {
      if (graph) {
        graph._destructor();
      }
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, []);

  useEffect(() => {
    if (!graphRef.current) return;

    const spin = () => {
      if (!rotating) return;

      const curRotation = graphRef.current.rotation();
      graphRef.current.rotation({
        x: curRotation.x,
        y: curRotation.y + 0.001,
        z: curRotation.z,
      });

      animationFrameId = requestAnimationFrame(spin);
    };

    let animationFrameId;
    if (rotating) {
      spin();
    }

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [rotating]);

  const handleZoomToFit = () => {
    if (graphRef.current) {
      graphRef.current.zoomToFit(400);
    }
  };

  const handleResetView = () => {
    if (graphRef.current) {
      graphRef.current.cameraPosition(
        { x: 0, y: 0, z: 300 },
        { x: 0, y: 0, z: 0 },
        1000
      );
      setSelectedNode(null);
    }
  };

  const handleToggleRotation = () => {
    setRotating(!rotating);
  };

  if (error) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-gray-900">
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 text-red-400">
          <p className="font-medium">Error loading graph visualization</p>
          <p className="text-sm mt-1">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full w-full relative">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-80 z-50">
          <div className="text-white p-5 rounded-lg">
            Loading project graph data...
          </div>
        </div>
      )}

      <div className="absolute top-4 left-4 bg-black bg-opacity-70 text-white p-4 rounded-lg z-10 max-w-xs">
        <h3 className="text-lg font-semibold mb-2">
          Project Graph Visualization
        </h3>
        <div className="space-x-2 mb-4">
          <button
            onClick={handleZoomToFit}
            className="bg-green-500 hover:bg-green-600 text-white px-3 py-1 rounded"
          >
            Zoom to Fit
          </button>
          <button
            onClick={handleResetView}
            className="bg-green-500 hover:bg-green-600 text-white px-3 py-1 rounded"
          >
            Reset View
          </button>
          <button
            onClick={handleToggleRotation}
            className="bg-green-500 hover:bg-green-600 text-white px-3 py-1 rounded"
          >
            {rotating ? 'Stop Rotation' : 'Start Rotation'}
          </button>
        </div>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(GROUP_COLORS).map(([group, color]) => (
            <div key={group} className="flex items-center">
              <div
                className="w-4 h-4 mr-2 rounded"
                style={{ backgroundColor: color }}
              />
              <span>{GROUP_FIELDS[group]}</span>
            </div>
          ))}
        </div>
        <p className="text-sm mt-2 text-gray-300">
          Click on nodes to view details. Links represent project similarity.
        </p>
      </div>

      {selectedNode && (
        <div className="absolute bottom-4 right-4 bg-black bg-opacity-70 text-white p-4 rounded-lg z-10 max-w-xs">
          <h3 className="text-lg font-semibold">{selectedNode.name}</h3>
          <p className="text-gray-300">{GROUP_FIELDS[selectedNode.group]}</p>
          <p className="mt-2">{selectedNode.description}</p>
        </div>
      )}

      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
};

export default GraphView;
