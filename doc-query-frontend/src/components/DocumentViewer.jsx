import { X } from 'lucide-react';

export default function DocumentViewer({ document, onClose, isDarkMode }) {
  if (!document) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className={`w-11/12 h-5/6 rounded-lg shadow-xl flex flex-col ${
        isDarkMode ? 'bg-gray-800' : 'bg-white'
      }`}>
        {/* Header */}
        <div className={`p-4 border-b ${
          isDarkMode ? 'border-gray-700' : 'border-gray-200'
        } flex justify-between items-center`}>
          <h2 className={`text-xl font-semibold ${
            isDarkMode ? 'text-white' : 'text-gray-800'
          }`}>
            {document.doc_name}
          </h2>
          <button
            onClick={onClose}
            className={`p-2 rounded-full hover:${
              isDarkMode ? 'bg-gray-700' : 'bg-gray-100'
            }`}
          >
            <X className={`h-5 w-5 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          <div className={`prose max-w-none ${
            isDarkMode ? 'prose-invert' : ''
          }`}>
            <div className="whitespace-pre-wrap">
              {document.original_content.split('\n').map((paragraph, index) => {
                // Check if this paragraph contains the highlighted text
                const isHighlighted = paragraph.includes(document.highlight_text_content);
                
                return (
                  <p
                    key={index}
                    className={`mb-4 ${
                      isHighlighted
                        ? 'bg-yellow-200 dark:bg-yellow-900/30 p-2 rounded'
                        : ''
                    }`}
                  >
                    {paragraph}
                  </p>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 