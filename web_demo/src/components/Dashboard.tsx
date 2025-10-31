import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

interface Assistant {
  assistant_id: string;
  name: string;
  status: string;
  created_at: string | null;
  deployment_ready: boolean;
}

export function Dashboard() {
  const [assistants, setAssistants] = useState<Assistant[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadAssistants();
  }, []);

  const loadAssistants = async () => {
    try {
      const response = await fetch('http://localhost:8080/assistants');
      if (!response.ok) throw new Error('Failed to load assistants');
      const data = await response.json();
      
      // Filter out system directories and invalid assistants
      const validAssistants = (data.assistants || []).filter((a: Assistant) => {
        const invalidIds = ['uploads', 'datasets', 'database', 'agents', 'test_assistant'];
        return !invalidIds.includes(a.assistant_id) && a.created_at !== null;
      });
      
      setAssistants(validAssistants);
      setError('');
    } catch (err) {
      setError('Backend not available. Start the API server on port 8080.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'bg-green-100 text-green-800';
      case 'training': return 'bg-yellow-100 text-yellow-800';
      case 'unknown': return 'bg-gray-100 text-gray-800';
      default: return 'bg-blue-100 text-blue-800';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold">🧠 CEREBROS Dashboard</h1>
          <nav className="flex gap-4">
            <Link to="/" className="hover:underline">Dashboard</Link>
            <Link to="/upload" className="hover:underline">Upload Data</Link>
            <Link to="/new" className="hover:underline">Training Wizard</Link>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto p-6">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-2">Your AI Assistants</h2>
          <p className="text-gray-600">Manage and interact with your personalized AI assistants</p>
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-6">
            <p className="font-semibold">⚠️ Connection Error</p>
            <p>{error}</p>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            <p className="mt-4 text-gray-600">Loading assistants...</p>
          </div>
        )}

        {/* Assistants Grid */}
        {!loading && !error && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {assistants.length === 0 ? (
              <div className="col-span-full text-center py-12 bg-white rounded-lg shadow">
                <p className="text-gray-500 text-lg mb-2">No assistants found</p>
                <p className="text-gray-400 text-sm mb-6">Create your first AI agent to get started</p>
                <Link 
                  to="/wizard" 
                  className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 font-semibold"
                >
                  Create Your First Agent
                </Link>
              </div>
            ) : (
              assistants.map((assistant) => (
                <div 
                  key={assistant.assistant_id} 
                  className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow"
                >
                  <div className="flex justify-between items-start mb-4">
                    <h3 className="text-xl font-semibold text-gray-900">
                      {assistant.name}
                    </h3>
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getStatusColor(assistant.status)}`}>
                      {assistant.status}
                    </span>
                  </div>
                  
                  <div className="space-y-2 mb-4">
                    <p className="text-sm text-gray-600">
                      <span className="font-medium">ID:</span> {assistant.assistant_id}
                    </p>
                    {assistant.created_at && (
                      <p className="text-sm text-gray-600">
                        <span className="font-medium">Created:</span> {new Date(assistant.created_at).toLocaleDateString()}
                      </p>
                    )}
                    <p className="text-sm text-gray-600">
                      <span className="font-medium">Ready:</span> {assistant.deployment_ready ? '✅ Yes' : '⏳ Training'}
                    </p>
                  </div>

                  <div className="flex gap-2">
                    {assistant.deployment_ready && (
                      <Link
                        to={`/chat/${assistant.assistant_id}`}
                        className="flex-1 bg-blue-600 text-white text-center px-4 py-2 rounded hover:bg-blue-700"
                      >
                        Chat
                      </Link>
                    )}
                    <Link
                      to={`/status/${assistant.assistant_id}`}
                      className="flex-1 bg-gray-200 text-gray-800 text-center px-4 py-2 rounded hover:bg-gray-300"
                    >
                      Details
                    </Link>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {/* Quick Stats */}
        {!loading && !error && assistants.length > 0 && (
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-white p-6 rounded-lg shadow">
              <p className="text-gray-600 text-sm">Total Assistants</p>
              <p className="text-3xl font-bold text-blue-600">{assistants.length}</p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow">
              <p className="text-gray-600 text-sm">Ready</p>
              <p className="text-3xl font-bold text-green-600">
                {assistants.filter(a => a.deployment_ready).length}
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow">
              <p className="text-gray-600 text-sm">Training</p>
              <p className="text-3xl font-bold text-yellow-600">
                {assistants.filter(a => !a.deployment_ready).length}
              </p>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
