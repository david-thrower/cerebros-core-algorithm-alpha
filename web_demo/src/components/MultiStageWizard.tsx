import React, { useState } from 'react';
import { Trash2Icon, CheckIcon, Edit2Icon, Upload, Loader2 } from 'lucide-react';

interface TrainingStage {
  stage: number;
  name: string;
  status: 'pending' | 'training' | 'complete' | 'error';
  progress: number;
  metrics?: {
    loss?: number;
    accuracy?: number;
    perplexity?: number;
  };
}

interface WizardStep {
  num: number;
  label: string;
  sublabel: string;
}

export function MultiStageWizard() {
  const [currentStep, setCurrentStep] = useState(1);
  const [agentId, setAgentId] = useState<string | null>(null);
  const [agentName, setAgentName] = useState('');
  const [trainingStatus, setTrainingStatus] = useState<'idle' | 'training' | 'complete' | 'error'>('idle');
  const [trainingStages, setTrainingStages] = useState<TrainingStage[]>([
    { stage: 1, name: 'Foundation', status: 'pending', progress: 0 },
    { stage: 2, name: 'Domain Adaptation', status: 'pending', progress: 0 },
    { stage: 3, name: 'Knowledge Integration', status: 'pending', progress: 0 },
    { stage: 4, name: 'Style Refinement', status: 'pending', progress: 0 },
    { stage: 5, name: 'Personalization', status: 'pending', progress: 0 },
  ]);

  const steps: WizardStep[] = [
    { num: 1, label: 'Work Products', sublabel: 'Upload example documents' },
    { num: 2, label: 'Prompts', sublabel: 'Add training examples' },
    { num: 3, label: 'Communications', sublabel: 'Upload conversations' },
    { num: 4, label: 'References', sublabel: 'Add knowledge base' },
    { num: 5, label: 'Training', sublabel: 'Build your assistant' },
  ];

  const API_URL = 'http://localhost:5000';

  const handleFileUpload = async (files: FileList, docType: string) => {
    if (!agentId) {
      // Create agent first
      const response = await fetch(`${API_URL}/api/agents`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: agentName || 'My Assistant' }),
      });
      const data = await response.json();
      setAgentId(data.agent_id);
      
      // Upload files
      const formData = new FormData();
      Array.from(files).forEach(file => formData.append('files', file));
      formData.append('type', docType);
      
      await fetch(`${API_URL}/api/agents/${data.agent_id}/documents`, {
        method: 'POST',
        body: formData,
      });
    } else {
      // Upload to existing agent
      const formData = new FormData();
      Array.from(files).forEach(file => formData.append('files', file));
      formData.append('type', docType);
      
      await fetch(`${API_URL}/api/agents/${agentId}/documents`, {
        method: 'POST',
        body: formData,
      });
    }
  };

  const startTraining = async () => {
    if (!agentId) return;
    
    setTrainingStatus('training');
    
    // Start training
    await fetch(`${API_URL}/api/agents/${agentId}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent_name: agentName }),
    });
    
    // Poll for status
    const pollInterval = setInterval(async () => {
      const response = await fetch(`${API_URL}/api/agents/${agentId}/status`);
      const status = await response.json();
      
      if (status.status === 'completed') {
        clearInterval(pollInterval);
        setTrainingStatus('complete');
        // Update all stages to complete
        setTrainingStages(prev => prev.map(s => ({ ...s, status: 'complete', progress: 100 })));
      } else if (status.status === 'failed') {
        clearInterval(pollInterval);
        setTrainingStatus('error');
      } else if (status.current_stage) {
        // Update stage progress
        setTrainingStages(prev => prev.map(s => ({
          ...s,
          status: s.stage < status.current_stage ? 'complete' : 
                  s.stage === status.current_stage ? 'training' : 'pending',
          progress: s.stage < status.current_stage ? 100 : 
                   s.stage === status.current_stage ? 50 : 0
        })));
      }
    }, 2000);
  };

  const renderStep = () => {
    switch(currentStep) {
      case 1:
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Upload Work Products</h2>
            <p className="text-gray-600">Upload example documents that represent your work style</p>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <input
                type="file"
                multiple
                accept=".txt,.pdf,.doc,.docx"
                onChange={(e) => e.target.files && handleFileUpload(e.target.files, 'work_product')}
                className="hidden"
                id="work-products"
              />
              <label htmlFor="work-products" className="cursor-pointer text-blue-600 hover:text-blue-700">
                Click to upload work products
              </label>
              <p className="text-sm text-gray-500 mt-2">PDF, DOC, DOCX, TXT up to 10MB each</p>
            </div>
          </div>
        );
      
      case 2:
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Add Training Examples</h2>
            <p className="text-gray-600">Provide example prompts and desired responses</p>
            {/* Reuse PromptTraining component structure */}
            <div className="bg-blue-50 p-4 rounded">
              <p className="text-sm">
                Add prompt-response pairs that demonstrate how your assistant should respond
              </p>
            </div>
          </div>
        );
      
      case 3:
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Upload Communications</h2>
            <p className="text-gray-600">Upload example emails or chat conversations</p>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <input
                type="file"
                multiple
                onChange={(e) => e.target.files && handleFileUpload(e.target.files, 'communication')}
                className="hidden"
                id="communications"
              />
              <label htmlFor="communications" className="cursor-pointer text-blue-600 hover:text-blue-700">
                Click to upload communications
              </label>
            </div>
          </div>
        );
      
      case 4:
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Add Reference Materials</h2>
            <p className="text-gray-600">Upload reference documents for your knowledge base</p>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
              <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <input
                type="file"
                multiple
                onChange={(e) => e.target.files && handleFileUpload(e.target.files, 'reference')}
                className="hidden"
                id="references"
              />
              <label htmlFor="references" className="cursor-pointer text-blue-600 hover:text-blue-700">
                Click to upload references
              </label>
            </div>
          </div>
        );
      
      case 5:
        return (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold">Multi-Stage Training Pipeline</h2>
            <p className="text-gray-600">Your assistant will be trained through 5 stages</p>
            
            {trainingStatus === 'idle' && (
              <div className="space-y-4">
                <div className="bg-white p-6 rounded-lg border">
                  <h3 className="font-semibold mb-4">Training Stages:</h3>
                  <ul className="space-y-2 text-sm">
                    <li>• Stage 1: Initial Foundation</li>
                    <li>• Stage 2: Domain Adaptation</li>
                    <li>• Stage 3: Knowledge Integration</li>
                    <li>• Stage 4: Style Refinement</li>
                    <li>• Stage 5: Personalization Fine-Tuning</li>
                  </ul>
                </div>
                <button
                  onClick={startTraining}
                  className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 font-semibold"
                >
                  Start Training
                </button>
              </div>
            )}
            
            {trainingStatus === 'training' && (
              <div className="space-y-4">
                {trainingStages.map(stage => (
                  <div key={stage.stage} className="bg-white p-4 rounded-lg border">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-3">
                        {stage.status === 'complete' && (
                          <CheckIcon className="w-5 h-5 text-green-600" />
                        )}
                        {stage.status === 'training' && (
                          <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
                        )}
                        {stage.status === 'pending' && (
                          <div className="w-5 h-5 rounded-full border-2 border-gray-300" />
                        )}
                        <span className="font-semibold">
                          Stage {stage.stage}: {stage.name}
                        </span>
                      </div>
                      <span className="text-sm text-gray-600">{stage.progress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all ${
                          stage.status === 'complete' ? 'bg-green-600' :
                          stage.status === 'training' ? 'bg-blue-600' : 'bg-gray-300'
                        }`}
                        style={{ width: `${stage.progress}%` }}
                      />
                    </div>
                    {stage.metrics && (
                      <div className="mt-2 text-xs text-gray-600">
                        Loss: {stage.metrics.loss?.toFixed(4)} | 
                        Accuracy: {(stage.metrics.accuracy! * 100).toFixed(2)}%
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
            
            {trainingStatus === 'complete' && (
              <div className="space-y-4">
                <div className="bg-green-50 border-2 border-green-600 rounded-lg p-6 text-center">
                  <CheckIcon className="w-16 h-16 text-green-600 mx-auto mb-4" />
                  <h3 className="text-2xl font-bold text-green-900 mb-2">Training Complete!</h3>
                  <p className="text-green-700">Your personalized assistant is ready to deploy</p>
                </div>
                <button
                  onClick={() => window.location.href = '/chat'}
                  className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 font-semibold"
                >
                  Start Using Your Assistant
                </button>
              </div>
            )}
          </div>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="w-full min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Create Your AI Assistant
          </h1>
          <input
            type="text"
            value={agentName}
            onChange={(e) => setAgentName(e.target.value)}
            placeholder="Enter assistant name..."
            className="text-lg text-gray-600 border-b-2 border-transparent hover:border-gray-300 focus:border-blue-600 focus:outline-none"
          />
        </div>

        {/* Step Navigation */}
        <div className="flex items-center justify-between mb-12">
          {steps.map((step, idx) => (
            <div key={step.num} className="flex items-center flex-1">
              <div className="flex flex-col items-center">
                <div
                  className={`w-12 h-12 rounded-full flex items-center justify-center text-white font-semibold cursor-pointer ${
                    step.num === currentStep
                      ? 'bg-blue-600'
                      : step.num < currentStep
                      ? 'bg-green-600'
                      : 'bg-gray-300'
                  }`}
                  onClick={() => setCurrentStep(step.num)}
                >
                  {step.num < currentStep ? <CheckIcon className="w-6 h-6" /> : step.num}
                </div>
                <div className="mt-2 text-center">
                  <div className="font-semibold text-sm text-gray-900">{step.label}</div>
                  <div className="text-xs text-gray-500">{step.sublabel}</div>
                </div>
              </div>
              {idx < steps.length - 1 && (
                <div className="flex-1 h-0.5 bg-gray-300 mx-4 mt-[-40px]" />
              )}
            </div>
          ))}
        </div>

        {/* Main Content */}
        <div className="bg-white rounded-lg shadow-sm p-8">
          {renderStep()}

          {/* Footer */}
          <div className="flex justify-between items-center mt-8 pt-6 border-t border-gray-200">
            <button
              onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
              disabled={currentStep === 1}
              className="px-6 py-2 text-gray-600 hover:text-gray-900 disabled:opacity-50"
            >
              Back
            </button>
            {currentStep < 5 && (
              <button
                onClick={() => setCurrentStep(Math.min(5, currentStep + 1))}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Continue
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
