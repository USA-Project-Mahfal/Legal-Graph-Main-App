import React, { useEffect, useRef, useState } from 'react';
import ForceGraph3D from '3d-force-graph';
import * as THREE from 'three';

const GraphView = () => {
  const containerRef = useRef(null);
  const graphRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [rotating, setRotating] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [error, setError] = useState(null);
  const [fieldColors, setFieldColors] = useState({});

  // Fetch field colors configuration
  useEffect(() => {
    const fetchFieldColors = async () => {
      try {
        const response = await fetch('http://localhost:8000/group-config');
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const colors = await response.json();
        setFieldColors(colors);
      } catch (err) {
        console.error('Error fetching field colors:', err);
        setError('Failed to load field colors configuration');
      }
    };
    fetchFieldColors();
  }, []);

  // Fetch graph data
  const fetchGraphData = async () => {
    try {
      const response = await fetch('http://localhost:8000/graph');
      if (!response.ok) {
        if (response.status === 500) {
          const errorData = await response.json();
          throw new Error(
            errorData.message || `HTTP error! Status: ${response.status}`
          );
        }
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return await response.json();
    } catch (err) {
      console.error('Error fetching graph data:', err);
      throw err;
    }
  };

  useEffect(() => {
    let graph = null;
    let animationFrameId = null;

    const initGraph = async () => {
      try {
        if (!containerRef.current) return;

        // Initialize the 3D force graph
        graph = ForceGraph3D({ controlType: 'orbit' })(containerRef.current)
          .backgroundColor('#111827')
          .nodeColor((node) => fieldColors[node.group] || '#808080')
          .nodeLabel((node) => `${node.name} (${node.group})`)
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
        const data = await fetchGraphData();
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
        setError(err.message || `HTTP error! Status: ${err.status}`);
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
  }, [fieldColors]); // Add fieldColors as dependency

  useEffect(() => {
    if (!graphRef.current) return;

    const spin = () => {
      if (!rotating) return;

      graphRef.current.scene().rotation.y += 0.01;

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
    // Determine if this is the no embeddings error
    const isNoEmbeddingsError = error.includes('No embeddings available');

    return (
      <div className="h-full w-full flex items-center justify-center bg-gray-900">
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-6 text-red-400 max-w-md">
          <p className="font-medium text-lg">
            Error loading graph visualization
          </p>
          <p className="text-sm mt-2">{error}</p>

          {isNoEmbeddingsError && (
            <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded text-blue-400 text-sm">
              <p className="font-medium">To get started:</p>
              <ol className="list-decimal ml-5 mt-2 space-y-1">
                <li>Upload document files using the upload function</li>
                <li>The system will generate embeddings for your documents</li>
                <li>Return to this view to see the document graph</li>
              </ol>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="h-full w-full relative">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/90 backdrop-blur-sm z-50">
          <div className="text-white p-5 rounded-lg bg-gray-800/50 border border-gray-700">
            <div className="animate-pulse">Loading project graph data...</div>
          </div>
        </div>
      )}

      <div className="absolute top-4 left-4 bg-gray-900/90 backdrop-blur-sm text-white p-4 rounded-lg z-10 max-w-xs border border-gray-700 shadow-xl">
        <h3 className="text-lg font-semibold mb-3 text-gray-100">
          Project Graph Visualization
        </h3>
        <div className="flex flex-wrap gap-2 mb-4">
          <button
            onClick={handleZoomToFit}
            className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-md text-sm font-medium transition-colors"
          >
            Zoom to Fit
          </button>
          <button
            onClick={handleResetView}
            className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-md text-sm font-medium transition-colors"
          >
            Reset View
          </button>
          <button
            onClick={handleToggleRotation}
            className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-md text-sm font-medium transition-colors"
          >
            {rotating ? 'Stop Rotation' : 'Start Rotation'}
          </button>
        </div>
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(fieldColors).map(([field, color]) => (
            <div key={field} className="flex items-center">
              <div
                className="w-3 h-3 mr-2 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-sm text-gray-300">{field}</span>
            </div>
          ))}
        </div>
        <p className="text-sm mt-3 text-gray-400">
          Click on nodes to view details. Links represent project similarity.
        </p>
      </div>

      {selectedNode && (
        <div className="absolute bottom-4 right-4 bg-gray-900/90 backdrop-blur-sm text-white p-4 rounded-lg z-10 max-w-xs border border-gray-700 shadow-xl">
          <h3 className="text-lg font-semibold text-gray-100">
            {selectedNode.name}
          </h3>
          <p className="text-sm text-gray-400">{selectedNode.group}</p>
          <p className="mt-2 text-sm text-gray-300">
            {selectedNode.description}
          </p>
        </div>
      )}

      <div ref={containerRef} className="w-full h-full" />
    </div>
  );
};

export default GraphView;
