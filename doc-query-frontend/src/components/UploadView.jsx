import { useState } from 'react';
import { Upload, Info } from 'lucide-react';
import FileItem from './upload/FileItem';

export default function UploadView({
  uploadedFiles,
  uploadProgress,
  handleFileUpload,
  removeFile,
}) {
  const [dragActive, setDragActive] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  // Handle file upload to backend
  const uploadToBackend = async (files) => {
    const formData = new FormData();
    for (const file of files) {
      formData.append('files', file);
    }

    try {
      const response = await fetch('http://localhost:8000/upload-multiple', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Upload failed');
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Upload error:', error);
      setUploadError(error.message);
      throw error;
    }
  };

  // Handle drag and drop
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    setUploadError(null);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      try {
        await uploadToBackend(e.dataTransfer.files);
        handleFileUpload(e.dataTransfer.files);
      } catch (error) {
        console.error('Error uploading files:', error);
      }
    }
  };

  const handleFileSelect = async (e) => {
    setUploadError(null);
    if (e.target.files && e.target.files.length > 0) {
      try {
        await uploadToBackend(e.target.files);
        handleFileUpload(e.target.files);
      } catch (error) {
        console.error('Error uploading files:', error);
      }
    }
  };

  return (
    <div className="w-full p-6 overflow-y-auto">
      <div className="max-w-3xl mx-auto">
        <div className="mb-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-2">
            Upload Legal Documents
          </h2>
          <p className="text-gray-600">
            Upload your legal documents for analysis. We support PDF and DOCX
            formats.
          </p>
        </div>

        {uploadError && (
          <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {uploadError}
          </div>
        )}

        {/* File upload area */}
        <div
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            dragActive
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-blue-400'
          }`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-upload').click()}
        >
          <Upload className="h-12 w-12 mx-auto text-blue-500 mb-4" />
          <h3 className="text-lg font-medium text-gray-700 mb-1">
            Drag & Drop Files Here
          </h3>
          <p className="text-gray-500 mb-4">or click to browse your files</p>
          <input
            id="file-upload"
            type="file"
            multiple
            accept=".pdf,.docx,.doc"
            className="hidden"
            onChange={handleFileSelect}
          />
          <div className="inline-flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            Select Files
          </div>
        </div>

        {/* Uploaded files list */}
        {uploadedFiles.length > 0 && (
          <div className="mt-8">
            <h3 className="text-lg font-medium text-gray-800 mb-4">
              Uploaded Documents
            </h3>
            <div className="space-y-3">
              {uploadedFiles.map((file) => (
                <FileItem
                  key={file.id}
                  file={file}
                  progress={uploadProgress[file.id] || 0}
                  onRemove={() => removeFile(file.id)}
                />
              ))}
            </div>

            {/* Processing info */}
            <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start">
              <Info className="h-5 w-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0" />
              <div>
                <p className="text-sm text-blue-800">
                  Your documents are being processed and indexed. This may take
                  a few minutes depending on the document size.
                </p>
                <p className="text-sm text-blue-700 mt-1">
                  Once processed, you can ask questions about them in the chat.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
