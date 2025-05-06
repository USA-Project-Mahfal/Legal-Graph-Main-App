import { MessageSquare, Clock } from "lucide-react";

export default function HistoryView({
  chatHistory,
  selectChat,
  createNewChat,
}) {
  return (
    <div className="w-full p-6 overflow-y-auto">
      <div className="max-w-3xl mx-auto">
        <div className="mb-6 flex justify-between items-center">
          <div>
            <h2 className="text-xl font-semibold text-gray-800 mb-1">
              Chat History
            </h2>
            <p className="text-gray-600">
              View and resume your previous conversations
            </p>
          </div>
          <button
            onClick={createNewChat}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center"
          >
            <MessageSquare className="h-4 w-4 mr-2" />
            New Chat
          </button>
        </div>

        {/* Chat history list */}
        <div className="space-y-3">
          {chatHistory.map((chat) => (
            <div
              key={chat.id}
              onClick={() => selectChat(chat.id)}
              className="bg-white border rounded-lg p-4 hover:bg-gray-50 cursor-pointer shadow-sm transition-colors"
            >
              <div className="flex justify-between">
                <h3 className="font-medium text-gray-800">{chat.title}</h3>
                <div className="text-sm text-gray-500">{chat.date}</div>
              </div>
              <div className="flex items-center mt-2 text-sm text-gray-600">
                <MessageSquare className="h-4 w-4 mr-1" />
                <span>{chat.messages} messages</span>
              </div>
            </div>
          ))}
        </div>

        {chatHistory.length === 0 && (
          <div className="text-center py-12">
            <Clock className="h-12 w-12 mx-auto text-gray-400 mb-3" />
            <h3 className="text-lg font-medium text-gray-700 mb-1">
              No Chat History Yet
            </h3>
            <p className="text-gray-500">
              Start a new conversation to see it here
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
