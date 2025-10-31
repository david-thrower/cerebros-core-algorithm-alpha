import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Upload, FileText, CheckCircle, XCircle } from 'lucide-react';

interface UploadResponse {
  status: string;
  assistant_id: string;
  filename: string;
  path: string;
  size_bytes: number;
  message: string;
}

export function UploadWizard() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string>('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      const ext = selectedFile.name.toLowerCase();
      if (ext.endsWith('.csv') || ext.endsWith('.json')) {
        setFile(selectedFile);
        setError('');
        setResult(null);
      } else {
        setError('Please select a CSV or JSON file');
        setFile(null);
      }
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    setError('');
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8080/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data: UploadResponse = await response.json();
      setResult(data);
      setFile(null);
      // Reset file input
      const fileInput = document.getElementById('fileInput') as HTMLInputElement;
      if (fileInput) fileInput.value = '';
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed. Ensure backend is running on port 8080.');
      console.error('Upload error:', err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold">📤 Upload Training Data</h1>
          <nav className="flex gap-4">
            <Link to="/" className="hover:underline">Dashboard</Link>
            <Link to="/new" className="hover:underline">Upload</Link>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-2xl mx-auto p-6">
        <div className="bg-white rounded-lg shadow-md p-8">
          <div className="mb-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Data File</h2>
            <p className="text-gray-600">
              Upload CSV or JSON files containing training examples, work products, or communication samples
            </p>
          </div>

          {/* Upload Form */}
          <form onSubmit={handleUpload} className="space-y-6">
            <div>
              <label htmlFor="fileInput" className="block text-sm font-medium text-gray-700 mb-2">
                Select File (CSV or JSON)
              </label>
              <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md hover:border-blue-400 transition-colors">
                <div className="space-y-1 text-center">
                  <Upload className="mx-auto h-12 w-12 text-gray-400" />
                  <div className="flex text-sm text-gray-600">
                    <label
                      htmlFor="fileInput"
                      className="relative cursor-pointer bg-white rounded-md font-medium text-blue-600 hover:text-blue-500 focus-within:outline-none"
                    >
                      <span>Upload a file</span>
                      <input
                        id="fileInput"
                        name="file"
                        type="file"
                        accept=".csv,.json"
                        onChange={handleFileChange}
                        className="sr-only"
                        disabled={uploading}
                      />
                    </label>
                    <p className="pl-1">or drag and drop</p>
                  </div>
                  <p className="text-xs text-gray-500">CSV or JSON up to 10MB</p>
                </div>
              </div>
            </div>

            {/* Selected File Display */}
            {file && (
              <div className="flex items-center gap-2 p-3 bg-blue-50 rounded-lg">
                <FileText className="w-5 h-5 text-blue-600" />
                <span className="flex-1 text-sm text-gray-700">{file.name}</span>
                <span className="text-xs text-gray-500">
                  {(file.size / 1024).toFixed(2)} KB
                </span>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={!file || uploading}
              className="w-full bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-semibold transition-colors"
            >
              {uploading ? 'Uploading...' : 'Upload File'}
            </button>
          </form>

          {/* Success Message */}
          {result && (
            <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <h3 className="font-semibold text-green-900">Upload Successful!</h3>
                  <p className="text-sm text-green-700 mt-1">{result.message}</p>
                  <div className="mt-3 space-y-1 text-sm text-green-800">
                    <p><span className="font-medium">Assistant ID:</span> {result.assistant_id}</p>
                    <p><span className="font-medium">Filename:</span> {result.filename}</p>
                    <p><span className="font-medium">Size:</span> {(result.size_bytes / 1024).toFixed(2)} KB</p>
                  </div>
                  <Link
                    to="/"
                    className="mt-4 inline-block text-sm text-blue-600 hover:text-blue-700 font-medium"
                  >
                    ← Back to Dashboard
                  </Link>
                </div>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="flex items-start gap-3">
                <XCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-semibold text-red-900">Upload Failed</h3>
                  <p className="text-sm text-red-700 mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Info Section */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-900 mb-2">💡 What can I upload?</h3>
          <ul className="space-y-1 text-sm text-blue-800">
            <li>• <strong>CSV files:</strong> Training examples with columns for prompts, reasoning, and outputs</li>
            <li>• <strong>JSON files:</strong> Structured data with your communication style and work samples</li>
            <li>• <strong>Work products:</strong> Documents that showcase your writing style</li>
            <li>• <strong>Communication samples:</strong> Emails, messages, or other text examples</li>
          </ul>
        </div>
      </main>
    </div>
  );
}
