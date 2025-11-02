import { useEffect, useRef, useState } from 'react';
import { Terminal, X, Maximize2, Minimize2 } from 'lucide-react';

interface TrainingTerminalProps {
  assistantId: string;
  assistantName: string;
  onClose: () => void;
}

export default function TrainingTerminal({ assistantId, assistantName, onClose }: TrainingTerminalProps) {
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<'connecting' | 'training' | 'completed' | 'failed'>('connecting');
  const [isExpanded, setIsExpanded] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const terminalRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect to WebSocket
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:8080/ws/training/${assistantId}`;
    
    console.log('Connecting to WebSocket:', wsUrl);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setStatus('training');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'init') {
        // Receive initial logs
        setLogs(data.logs || []);
        setStatus(data.status === 'completed' ? 'completed' : 'training');
      } else if (data.type === 'log') {
        // Append new log line
        setLogs(prev => [...prev, data.data]);
      } else if (data.type === 'complete') {
        setStatus(data.status === 'success' ? 'completed' : 'failed');
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setStatus('failed');
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
    };

    // Cleanup on unmount
    return () => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    };
  }, [assistantId]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  const getStatusColor = () => {
    switch (status) {
      case 'connecting': return 'bg-yellow-500';
      case 'training': return 'bg-blue-500 animate-pulse';
      case 'completed': return 'bg-green-500';
      case 'failed': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'connecting': return 'Connecting...';
      case 'training': return 'Training in Progress';
      case 'completed': return 'Training Complete ✓';
      case 'failed': return 'Training Failed ✗';
      default: return 'Unknown';
    }
  };

  if (isMinimized) {
    // Minimized view - just a taskbar-style button
    return (
      <button
        onClick={() => setIsMinimized(false)}
        className="fixed bottom-4 right-4 bg-gray-800 hover:bg-gray-700 text-white px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 border-2 border-gray-600 transition z-50"
      >
        <Terminal className="w-5 h-5 text-green-400" />
        <div className="text-left">
          <div className="font-semibold text-sm">{assistantName}</div>
          <div className="text-xs text-gray-400">{getStatusText()}</div>
        </div>
        <div className={`w-2 h-2 rounded-full ${getStatusColor()}`}></div>
      </button>
    );
  }

  return (
    <div
      className={`fixed bg-gray-900 border-2 border-gray-700 rounded-lg shadow-2xl overflow-hidden flex flex-col z-50 transition-all duration-300 ${
        isExpanded
          ? 'inset-4'
          : 'bottom-4 right-4 w-[600px] h-[400px]'
      }`}
    >
      {/* Header */}
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between border-b border-gray-700">
        <div className="flex items-center gap-3">
          <Terminal className="w-5 h-5 text-green-400" />
          <div>
            <div className="text-white font-semibold">{assistantName}</div>
            <div className="text-xs text-gray-400">Training Terminal</div>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Status indicator */}
          <div className="flex items-center gap-2 px-3 py-1 bg-gray-700 rounded">
            <div className={`w-2 h-2 rounded-full ${getStatusColor()}`}></div>
            <span className="text-xs text-gray-300">{getStatusText()}</span>
          </div>
          
          {/* Controls */}
          <button
            onClick={() => setIsMinimized(true)}
            className="p-2 hover:bg-gray-700 rounded text-gray-400 hover:text-white transition"
            title="Minimize to taskbar"
          >
            <Minimize2 className="w-4 h-4" />
          </button>
          
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-2 hover:bg-gray-700 rounded text-gray-400 hover:text-white transition"
            title={isExpanded ? "Restore" : "Maximize"}
          >
            <Maximize2 className="w-4 h-4" />
          </button>
          
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-700 rounded text-gray-400 hover:text-red-400 transition"
            title="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Terminal content */}
      <div
        ref={terminalRef}
        className="flex-1 overflow-y-auto bg-black p-4 font-mono text-sm"
      >
        {logs.length === 0 ? (
          <div className="text-gray-500 italic">
            Waiting for training output...
          </div>
        ) : (
          logs.map((line, idx) => (
            <div key={idx} className="text-green-400 whitespace-pre-wrap break-words">
              {line}
            </div>
          ))
        )}
        
        {/* Cursor */}
        {status === 'training' && (
          <span className="inline-block w-2 h-4 bg-green-400 animate-pulse ml-1"></span>
        )}
      </div>

      {/* Footer with stats */}
      <div className="bg-gray-800 px-4 py-2 border-t border-gray-700 text-xs text-gray-400 flex justify-between">
        <span>Lines: {logs.length}</span>
        <span>Assistant ID: {assistantId}</span>
      </div>
    </div>
  );
}
