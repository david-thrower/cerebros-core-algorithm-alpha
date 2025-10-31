import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Upload, FileText, CheckCircle, ArrowRight, ArrowLeft, Loader } from 'lucide-react';

interface UploadedFile {
  name: string;
  size: number;
  stage: number;
}

interface ProcessResponse {
  status: string;
  assistant_id: string;
  stage: number;
  csv_file: string;
  chunks_created: number;
}

export function CreateAssistantWizard() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(0);
  const [assistantName, setAssistantName] = useState('');
  const [assistantId, setAssistantId] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<Record<number, UploadedFile[]>>({
    1: [],
    2: [],
    3: [],
    4: []
  });
  const [uploading, setUploading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState('');

  const steps = [
    { num: 0, label: 'Name', sublabel: 'Name your assistant' },
    { num: 1, label: 'Work Products', sublabel: 'Upload example documents' },
    { num: 2, label: 'Communications', sublabel: 'Upload emails/messages' },
    { num: 3, label: 'References', sublabel: 'Add knowledge base' },
    { num: 4, label: 'Review & Train', sublabel: 'Start training' }
  ];

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    setError('');

    try {
      const currentId = assistantId;

      for (const file of Array.from(files)) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('assistant_id', currentId);
        formData.append('stage', currentStep.toString());

        const response = await fetch('http://localhost:8080/api/process-stage', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error(`Failed to process ${file.name}`);
        }

        const data: ProcessResponse = await response.json();

        // Add to uploaded files list
        setUploadedFiles(prev => ({
          ...prev,
          [currentStep]: [
            ...prev[currentStep],
            { name: file.name, size: file.size, stage: currentStep }
          ]
        }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleNext = () => {
    // Generate assistant ID from name on first step
    if (currentStep === 0 && !assistantId) {
      const sanitizedName = assistantName.toLowerCase().replace(/[^a-z0-9]/g, '_');
      const newId = sanitizedName || `assistant_${Date.now()}`;
      setAssistantId(newId);
    }
    
    if (currentStep === 4) {
      startTraining();
    } else {
      setCurrentStep(prev => Math.min(4, prev + 1));
    }
  };

  const handleBack = () => {
    setCurrentStep(prev => Math.max(0, prev - 1));
  };

  const startTraining = async () => {
    if (!assistantId) {
      setError('No assistant ID found');
      return;
    }

    setProcessing(true);
    setError('');

    try {
      console.log('Starting training for:', assistantId, assistantName);
      console.log('Total files uploaded:', getTotalFiles());
      
      const response = await fetch('http://localhost:8080/assistants/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          assistant_id: assistantId,
          assistant_name: assistantName || assistantId
        })
      });

      console.log('Response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Training failed:', response.status, errorText);
        throw new Error(`Training failed: ${response.status}`);
      }

      const data = await response.json();
      console.log('Training started successfully:', data);
      
      // Keep processing state true to show success message
      // Navigate to dashboard after delay
      setTimeout(() => {
        console.log('Navigating to dashboard');
        navigate('/');
      }, 3000);
    } catch (err) {
      console.error('Error in startTraining:', err);
      setError(err instanceof Error ? err.message : 'Training failed');
      setProcessing(false);
    }
  };

  const canProceed = () => {
    if (currentStep === 0) {
      return assistantName.trim().length > 0;
    }
    if (currentStep === 4) {
      return getTotalFiles() > 0;
    }
    return uploadedFiles[currentStep]?.length > 0;
  };

  const getTotalFiles = () => {
    return Object.values(uploadedFiles).reduce((sum, files) => sum + files.length, 0);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold">🧠 Create AI Assistant</h1>
          <Link to="/" className="text-white hover:underline">← Cancel</Link>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-5xl mx-auto p-6">

        {/* Step Indicators */}
        <div className="flex items-center justify-between mb-8">
          {steps.map((step, idx) => (
            <div key={step.num} className="flex items-center flex-1">
              <div className="flex flex-col items-center">
                <div className={`w-12 h-12 rounded-full flex items-center justify-center text-white font-semibold ${
                  step.num === currentStep ? 'bg-blue-600' : 
                  step.num < currentStep ? 'bg-green-500' : 'bg-gray-300'
                }`}>
                  {step.num < currentStep ? <CheckCircle className="w-6 h-6" /> : step.num}
                </div>
                <div className="mt-2 text-center">
                  <div className="font-semibold text-sm text-gray-900">{step.label}</div>
                  <div className="text-xs text-gray-500">{step.sublabel}</div>
                </div>
              </div>
              {idx < steps.length - 1 && (
                <div className={`flex-1 h-0.5 mx-4 mt-[-40px] ${
                  step.num < currentStep ? 'bg-green-500' : 'bg-gray-300'
                }`} />
              )}
            </div>
          ))}
        </div>

        {/* Name Input Step */}
        {currentStep === 0 && (
          <div className="bg-white rounded-lg shadow-md p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Name Your Assistant
            </h2>
            <p className="text-gray-600 mb-6">
              Give your AI assistant a memorable name
            </p>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Assistant Name
            </label>
            <input
              type="text"
              value={assistantName}
              onChange={(e) => setAssistantName(e.target.value)}
              placeholder="My Personal Assistant"
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-lg"
              autoFocus
            />
            {assistantName && (
              <p className="mt-3 text-sm text-green-600 flex items-center gap-2">
                <CheckCircle className="w-4 h-4" />
                Assistant ID will be: {assistantName.toLowerCase().replace(/[^a-z0-9]/g, '_')}
              </p>
            )}
          </div>
        )}

        {/* Upload Area */}
        {currentStep > 0 && currentStep < 4 && (
          <div className="bg-white rounded-lg shadow-md p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              {steps[currentStep].label}
            </h2>
            <p className="text-gray-600 mb-6">
              {currentStep === 1 && 'Upload documents that showcase your work style (PDFs, Word docs, text files)'}
              {currentStep === 2 && 'Upload emails, messages, or communication samples (TXT, CSV, JSON)'}
              {currentStep === 3 && 'Upload reference materials or knowledge base documents'}
            </p>

            {/* File Upload */}
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors">
              <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <label className="cursor-pointer">
                <span className="text-blue-600 hover:text-blue-700 font-medium">
                  Click to upload
                </span>
                <span className="text-gray-600"> or drag and drop</span>
                <input
                  type="file"
                  multiple
                  accept=".txt,.csv,.json,.pdf,.doc,.docx"
                  onChange={handleFileUpload}
                  className="hidden"
                  disabled={uploading}
                />
              </label>
              <p className="text-xs text-gray-500 mt-2">
                TXT, CSV, JSON, PDF, DOC, DOCX (max 10MB each)
              </p>
            </div>

            {/* Uploaded Files List */}
            {uploadedFiles[currentStep].length > 0 && (
              <div className="mt-6">
                <h3 className="font-semibold text-gray-900 mb-3">
                  Uploaded Files ({uploadedFiles[currentStep].length})
                </h3>
                <div className="space-y-2">
                  {uploadedFiles[currentStep].map((file, idx) => (
                    <div key={idx} className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
                      <FileText className="w-5 h-5 text-green-600" />
                      <span className="flex-1 text-sm text-gray-700">{file.name}</span>
                      <span className="text-xs text-gray-500">
                        {(file.size / 1024).toFixed(2)} KB
                      </span>
                      <CheckCircle className="w-5 h-5 text-green-600" />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {uploading && (
              <div className="mt-4 flex items-center gap-2 text-blue-600">
                <Loader className="w-5 h-5 animate-spin" />
                <span>Processing files...</span>
              </div>
            )}
          </div>
        )}

        {/* Review Step */}
        {currentStep === 4 && (
          <div className="bg-white rounded-lg shadow-md p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Review & Start Training</h2>
            <p className="text-gray-600 mb-6">
              Review your uploaded data and start training your assistant
            </p>

            <div className="space-y-4 mb-6">
              <div className="p-4 bg-blue-50 rounded-lg">
                <p className="font-semibold text-blue-900">Assistant Name</p>
                <p className="text-blue-700">{assistantName || assistantId}</p>
              </div>

              <div className="p-4 bg-gray-50 rounded-lg">
                <p className="font-semibold text-gray-900 mb-3">Uploaded Data Summary</p>
                {steps.slice(1, 4).map(step => (
                  <div key={step.num} className="flex justify-between py-2 border-b border-gray-200">
                    <span className="text-gray-700">{step.label}</span>
                    <span className="font-semibold text-gray-900">
                      {uploadedFiles[step.num].length} files
                    </span>
                  </div>
                ))}
                <div className="flex justify-between py-2 font-bold">
                  <span>Total Files</span>
                  <span className="text-blue-600">{getTotalFiles()}</span>
                </div>
              </div>
            </div>

            {processing && (
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg mb-6">
                <div className="flex items-center gap-3">
                  <Loader className="w-5 h-5 text-green-600 animate-spin" />
                  <div>
                    <p className="font-semibold text-green-900">Training Started!</p>
                    <p className="text-sm text-green-700">
                      Assistant "{assistantName || assistantId}" is being trained with {getTotalFiles()} files.
                    </p>
                    <p className="text-xs text-green-600 mt-1">
                      Redirecting to dashboard in 3 seconds...
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-700">{error}</p>
          </div>
        )}

        {/* Navigation Buttons */}
        <div className="flex justify-between mt-8">
          <button
            onClick={handleBack}
            disabled={currentStep === 0 || processing}
            className="flex items-center gap-2 px-6 py-3 text-gray-600 hover:text-gray-900 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ArrowLeft className="w-5 h-5" />
            Back
          </button>

          <button
            onClick={handleNext}
            disabled={!canProceed() || uploading || processing}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-semibold"
          >
            {currentStep === 4 ? (
              processing ? (
                <>
                  <Loader className="w-5 h-5 animate-spin" />
                  Training...
                </>
              ) : (
                <>
                  Start Training
                  <CheckCircle className="w-5 h-5" />
                </>
              )
            ) : (
              <>
                Continue
                <ArrowRight className="w-5 h-5" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
