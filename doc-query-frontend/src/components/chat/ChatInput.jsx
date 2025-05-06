import { useState } from "react";
import { Send, Paperclip, Smile, CornerDownLeft } from "lucide-react";

export default function ChatInput({
  input,
  setInput,
  handleSubmit,
  handleKeyDown,
  isDarkMode,
}) {
  const [isFocused, setIsFocused] = useState(false);

  return (
    <div
      className={`p-3 border-t ${
        isDarkMode
          ? "border-gray-700 bg-gray-800"
          : "border-gray-200 bg-gray-50"
      }`}
    >
      <div className="max-w-4xl mx-auto">
        <form onSubmit={handleSubmit}>
          <div
            className={`flex items-center rounded-xl ${
              isDarkMode ? "bg-gray-700" : "bg-white"
            } py-2 px-3 shadow-sm ${
              isFocused
                ? isDarkMode
                  ? "ring-1 ring-blue-500 border-gray-600"
                  : "ring-2 ring-blue-100 border-blue-300"
                : isDarkMode
                ? "border-gray-600"
                : "border-gray-300"
            } border transition-all`}
          >
            {/* File attachment button (visual only) */}
            <button
              type="button"
              className={`p-1.5 rounded-full ${
                isDarkMode
                  ? "hover:bg-gray-600 text-gray-300"
                  : "hover:bg-gray-100 text-gray-500"
              }`}
            >
              <Paperclip className="h-4 w-4" />
            </button>

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="Ask about your legal documents..."
              className={`flex-grow mx-2 py-1 px-1 outline-none resize-none max-h-32 text-sm ${
                isDarkMode
                  ? "bg-gray-700 text-white placeholder-gray-400"
                  : "bg-white text-gray-900 placeholder-gray-500"
              }`}
              rows="1"
            />

            {/* Emoji button (visual only) */}
            <button
              type="button"
              className={`p-1.5 rounded-full mr-1 ${
                isDarkMode
                  ? "hover:bg-gray-600 text-gray-300"
                  : "hover:bg-gray-100 text-gray-500"
              }`}
            >
              <Smile className="h-4 w-4" />
            </button>

            <button
              type="submit"
              disabled={!input.trim() || input.length === 0}
              className={`p-2 rounded-full min-w-[36px] flex items-center justify-center ${
                input.trim() && input.length > 0
                  ? "bg-blue-600 hover:bg-blue-700 text-white"
                  : isDarkMode
                  ? "bg-gray-600 text-gray-400 cursor-not-allowed"
                  : "bg-gray-200 text-gray-500 cursor-not-allowed"
              } transition-colors`}
            >
              <Send className="h-4 w-4" />
            </button>
          </div>

          <div className="flex items-center justify-center mt-2">
            <div
              className={`text-xs flex items-center ${
                isDarkMode ? "text-gray-400" : "text-gray-500"
              }`}
            >
              <CornerDownLeft className="h-3 w-3 mr-1" />
              <span>Press Enter to send, Shift+Enter for a new line</span>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
