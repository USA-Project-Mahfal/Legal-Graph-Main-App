import { Bot, User, AlertCircle, FileText, ExternalLink } from "lucide-react";

export default function MessageList({
  messages,
  isLoading,
  messagesEndRef,
  isDarkMode,
  onViewDocument,
}) {
  // Function to format message timestamp
  const getFormattedTime = () => {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div className="flex-grow overflow-y-auto px-1 py-0 scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-transparent">
      <div className="max-w-5xl mx-auto space-y-2">
        {/* Today date indicator - reduced top/bottom margin */}
        <div className="flex justify-center my-2">
          <div
            className={`px-3 py-1 rounded-full text-xs ${
              isDarkMode
                ? "bg-gray-700 text-gray-300"
                : "bg-gray-100 text-gray-600"
            }`}
          >
            Today, {new Date().toLocaleDateString()}
          </div>
        </div>

        {messages.map((message, index) => (
          <div
            key={message.id}
            className={`flex ${
              message.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`relative max-w-[85%] rounded-2xl px-3 py-1.5 shadow-sm ${
                message.role === "user"
                  ? "bg-blue-600 text-white"
                  : isDarkMode
                  ? "bg-gray-700 text-gray-200"
                  : "bg-gray-100 text-gray-800"
              } ${
                index === 0 || messages[index - 1].role !== message.role
                  ? "mt-2"
                  : "mt-1"
              }`}
            >
              {/* Message header - Show only for first message in a sequence */}
              {(index === 0 || messages[index - 1].role !== message.role) && (
                <div className="flex items-center mb-0.5">
                  <div
                    className={`p-0.5 rounded-full ${
                      message.role === "user"
                        ? "bg-blue-500"
                        : isDarkMode
                        ? "bg-gray-600"
                        : "bg-gray-200"
                    }`}
                  >
                    {message.role === "user" ? (
                      <User className="h-2.5 w-2.5 text-white" />
                    ) : (
                      <Bot className="h-2.5 w-2.5 text-blue-400" />
                    )}
                  </div>
                  <div className="ml-1 font-medium text-xs">
                    {message.role === "user" ? "You" : "Legal Assistant"}
                  </div>
                  <div
                    className={`text-[10px] ml-auto ${
                      message.role === "user"
                        ? "text-blue-200"
                        : isDarkMode
                        ? "text-gray-400"
                        : "text-gray-500"
                    }`}
                  >
                    {getFormattedTime()}
                  </div>
                </div>
              )}

              {/* Message content */}
              <div className="whitespace-pre-wrap text-sm break-words">
                {message.content}
              </div>

              {/* Source citations - reduced spacing */}
              {message.sources && message.sources.length > 0 && (
                <div
                  className={`mt-1.5 p-1.5 rounded-md text-xs ${
                    message.role === "user"
                      ? "bg-blue-500"
                      : isDarkMode
                      ? "bg-gray-600"
                      : "bg-gray-200"
                  }`}
                >
                  <div className="flex items-center mb-0.5">
                    <AlertCircle className="h-2.5 w-2.5 mr-1" />
                    <p className="font-semibold">Sources:</p>
                  </div>
                  <ul className="space-y-0.5">
                    {message.sources.map((source, idx) => (
                      <li key={idx} className="flex items-center">
                        <FileText className="h-2.5 w-2.5 mr-1 flex-shrink-0" />
                        <div className="flex flex-col flex-grow min-w-0">
                          <div className="flex items-center">
                            <span className="font-medium truncate">
                              {source.title}
                            </span>
                            <div
                              className={`ml-auto text-[10px] px-1 py-0.5 rounded ${
                                isDarkMode
                                  ? "bg-gray-700 text-gray-300"
                                  : "bg-gray-300 text-gray-700"
                              }`}
                            >
                              {Math.round(source.relevance * 100)}%
                            </div>
                          </div>
                          <span className="opacity-90 text-[10px]">
                            {source.section}
                          </span>
                        </div>
                        <button
                          className={`ml-1 p-0.5 rounded hover:${
                            isDarkMode ? "bg-gray-700" : "bg-gray-300"
                          } flex-shrink-0`}
                        >
                          <ExternalLink className="h-2 w-2" />
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Chat bubble arrow - only if it's not a consecutive message from the same role */}
              {(index === 0 || messages[index - 1].role !== message.role) && (
                <div
                  className={`absolute top-2.5 ${
                    message.role === "user"
                      ? "right-[-5px] border-l-blue-600"
                      : "left-[-5px] border-r-[7px] " +
                        (isDarkMode ? "border-r-gray-700" : "border-r-gray-100")
                  } border-y-transparent border-y-[3.5px] ${
                    message.role === "user" ? "border-l-[7px]" : "border-l-0"
                  } border-r-0`}
                ></div>
              )}

              {/* Add document view button if there's a highlighted document */}
              {message.highlighted_document && (
                <button
                  onClick={() => onViewDocument(message.highlighted_document)}
                  className={`mt-2 flex items-center space-x-1 text-sm ${
                    isDarkMode ? 'text-blue-300' : 'text-blue-600'
                  } hover:underline`}
                >
                  <FileText className="h-4 w-4" />
                  <span>View Document</span>
                </button>
              )}
            </div>
          </div>
        ))}

        {/* Loading indicator - reduced spacing */}
        {isLoading && (
          <div className="flex justify-start">
            <div
              className={`relative max-w-[85%] rounded-2xl px-3 py-1.5 shadow-sm ${
                isDarkMode ? "bg-gray-700" : "bg-gray-100"
              }`}
            >
              <div className="flex items-center mb-0.5">
                <div
                  className={`p-0.5 rounded-full ${
                    isDarkMode ? "bg-gray-600" : "bg-gray-200"
                  }`}
                >
                  <Bot className="h-2.5 w-2.5 text-blue-400" />
                </div>
                <div
                  className={`ml-1 font-medium text-xs ${
                    isDarkMode ? "text-gray-200" : "text-gray-800"
                  }`}
                >
                  Legal Assistant
                </div>
                <div
                  className={`text-[10px] ml-auto ${
                    isDarkMode ? "text-gray-400" : "text-gray-500"
                  }`}
                >
                  {getFormattedTime()}
                </div>
              </div>
              <div className="flex space-x-1.5 py-0.5">
                <div className="h-1.5 w-1.5 bg-blue-400 rounded-full animate-bounce"></div>
                <div
                  className="h-1.5 w-1.5 bg-blue-400 rounded-full animate-bounce"
                  style={{ animationDelay: "0.2s" }}
                ></div>
                <div
                  className="h-1.5 w-1.5 bg-blue-400 rounded-full animate-bounce"
                  style={{ animationDelay: "0.4s" }}
                ></div>
              </div>
            </div>
          </div>
        )}

        {/* Empty starting state - reduced spacing */}
        {messages.length === 1 && messages[0].role === "system" && (
          <div className="flex flex-col items-center justify-center text-center px-3 py-6">
            <div
              className={`p-2.5 rounded-full mb-3 ${
                isDarkMode ? "bg-gray-700" : "bg-gray-100"
              }`}
            >
              <Bot
                className={`h-5 w-5 ${
                  isDarkMode ? "text-blue-400" : "text-blue-600"
                }`}
              />
            </div>
            <h3
              className={`text-lg font-semibold mb-1.5 ${
                isDarkMode ? "text-white" : "text-gray-800"
              }`}
            >
              Legal Intelligence
            </h3>
            <p
              className={`text-sm max-w-md ${
                isDarkMode ? "text-gray-300" : "text-gray-600"
              }`}
            >
              Ask questions about your legal documents, and I'll provide answers
              based on the content you've uploaded.
            </p>
            <div className="grid grid-cols-2 gap-2 mt-4 w-full max-w-md">
              <div
                className={`p-2.5 rounded-lg ${
                  isDarkMode
                    ? "bg-gray-700 hover:bg-gray-600"
                    : "bg-gray-100 hover:bg-gray-200"
                } cursor-pointer transition-colors`}
              >
                <h4
                  className={`font-medium text-sm mb-0.5 ${
                    isDarkMode ? "text-white" : "text-gray-800"
                  }`}
                >
                  NDA Review
                </h4>
                <p
                  className={`text-xs ${
                    isDarkMode ? "text-gray-400" : "text-gray-500"
                  }`}
                >
                  "Can I share data with third parties under this NDA?"
                </p>
              </div>
              <div
                className={`p-2.5 rounded-lg ${
                  isDarkMode
                    ? "bg-gray-700 hover:bg-gray-600"
                    : "bg-gray-100 hover:bg-gray-200"
                } cursor-pointer transition-colors`}
              >
                <h4
                  className={`font-medium text-sm mb-0.5 ${
                    isDarkMode ? "text-white" : "text-gray-800"
                  }`}
                >
                  Contract Analysis
                </h4>
                <p
                  className={`text-xs ${
                    isDarkMode ? "text-gray-400" : "text-gray-500"
                  }`}
                >
                  "Summarize our termination obligations in section 12"
                </p>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}
