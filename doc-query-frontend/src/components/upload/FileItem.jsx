import { X } from 'lucide-react';

export default function FileItem({ file, progress, onRemove }) {
  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    else return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  // Render file icon based on file type
  const getFileIcon = (fileType) => {
    if (fileType.includes('pdf')) return '📄';
    if (fileType.includes('word') || fileType.includes('docx')) return '📝';
    return '📋';
  };

  return (
    <div className="bg-white border rounded-lg p-4 flex items-center shadow-sm">
      <div className="text-2xl mr-3">{getFileIcon(file.type)}</div>
      <div className="flex-grow">
        <div className="flex justify-between">
          <div className="font-medium text-gray-800 truncate">{file.name}</div>
          <div className="text-sm text-gray-500">
            {formatFileSize(file.size)}
          </div>
        </div>
        <div className="mt-2">
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-600 rounded-full"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <div className="mt-1 text-xs text-right text-gray-500">
            {progress === 100 ? 'Processed' : `${progress}% uploaded`}
          </div>
        </div>
      </div>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onRemove();
        }}
        className="ml-4 text-gray-400 hover:text-red-500"
      >
        <X className="h-5 w-5" />
      </button>
    </div>
  );
}
