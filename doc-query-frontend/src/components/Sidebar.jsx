import {
  MessageSquare,
  Upload,
  Clock,
  ChevronLeft,
  ChevronRight,
  Plus,
  User,
  BookOpen,
  FileText,
  Shield,
  Settings,
  LogOut,
  Network,
} from 'lucide-react';

export default function Sidebar({
  activeTab,
  setActiveTab,
  createNewChat,
  chatHistory,
  selectChat,
  isSidebarCollapsed,
  setIsSidebarCollapsed,
}) {
  return (
    <div
      className={`bg-gradient-to-b from-gray-900 to-gray-800 text-white transition-all duration-300 flex flex-col shadow-lg ${
        isSidebarCollapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Top section with logo and collapse button */}
      <div className="flex items-center p-4 border-b border-gray-700/50">
        {!isSidebarCollapsed && (
          <div className="flex items-center flex-grow">
            <div className="flex items-center justify-center rounded-md bg-gradient-to-br from-blue-500 to-blue-600 h-9 w-9 shadow-inner shadow-blue-400/10">
              <Shield className="h-5 w-5 text-white" />
            </div>
            <div className="ml-2.5">
              <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-white to-gray-300 text-transparent bg-clip-text">
                LegalAssist
              </h1>
              <div className="text-[9px] text-blue-400 -mt-1 font-medium">
                DOCUMENT INTELLIGENCE
              </div>
            </div>
          </div>
        )}
        {isSidebarCollapsed && (
          <div className="flex items-center justify-center rounded-md bg-gradient-to-br from-blue-500 to-blue-600 h-8 w-8 mx-auto shadow-inner shadow-blue-400/10">
            <Shield className="h-4 w-4 text-white" />
          </div>
        )}
        <button
          onClick={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
          className="text-gray-400 hover:text-white p-1.5 rounded-full hover:bg-gray-700/50 ml-2"
        >
          {isSidebarCollapsed ? (
            <ChevronRight size={16} />
          ) : (
            <ChevronLeft size={16} />
          )}
        </button>
      </div>

      {/* New chat button */}
      <button
        onClick={createNewChat}
        className="flex items-center mx-3 my-3 p-2 rounded-lg bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 transition-all shadow-md shadow-blue-900/20"
      >
        <Plus className="h-4 w-4" />
        {!isSidebarCollapsed && (
          <span className="ml-2 font-medium text-sm">New Conversation</span>
        )}
      </button>

      {/* Navigation menu */}
      <nav className="flex-grow py-3">
        <ul className="space-y-1 px-2">
          <li>
            <button
              onClick={() => setActiveTab('chat')}
              className={`w-full flex items-center p-2 rounded-lg transition-colors ${
                activeTab === 'chat'
                  ? 'bg-blue-600/10 text-blue-400'
                  : 'text-gray-300 hover:bg-gray-800/50'
              }`}
            >
              <MessageSquare
                className={`h-4 w-4 ${
                  activeTab === 'chat' ? 'text-blue-400' : ''
                }`}
              />
              {!isSidebarCollapsed && (
                <span className="ml-3 text-sm">Chat</span>
              )}
            </button>
          </li>
          <li>
            <button
              onClick={() => setActiveTab('upload')}
              className={`w-full flex items-center p-2 rounded-lg transition-colors ${
                activeTab === 'upload'
                  ? 'bg-blue-600/10 text-blue-400'
                  : 'text-gray-300 hover:bg-gray-800/50'
              }`}
            >
              <Upload
                className={`h-4 w-4 ${
                  activeTab === 'upload' ? 'text-blue-400' : ''
                }`}
              />
              {!isSidebarCollapsed && (
                <span className="ml-3 text-sm">Upload</span>
              )}
            </button>
          </li>
          <li>
            <button
              onClick={() => setActiveTab('graph')}
              className={`w-full flex items-center p-2 rounded-lg transition-colors ${
                activeTab === 'graph'
                  ? 'bg-blue-600/10 text-blue-400'
                  : 'text-gray-300 hover:bg-gray-800/50'
              }`}
            >
              <Network
                className={`h-4 w-4 ${
                  activeTab === 'graph' ? 'text-blue-400' : ''
                }`}
              />
              {!isSidebarCollapsed && (
                <span className="ml-3 text-sm">Graph View</span>
              )}
            </button>
          </li>
          <li>
            <button
              onClick={() => setActiveTab('history')}
              className={`w-full flex items-center p-2 rounded-lg transition-colors ${
                activeTab === 'history'
                  ? 'bg-blue-600/10 text-blue-400'
                  : 'text-gray-300 hover:bg-gray-800/50'
              }`}
            >
              <Clock
                className={`h-4 w-4 ${
                  activeTab === 'history' ? 'text-blue-400' : ''
                }`}
              />
              {!isSidebarCollapsed && (
                <span className="ml-3 text-sm">History</span>
              )}
            </button>
          </li>
        </ul>

        {/* Recent chats */}
        {!isSidebarCollapsed && (
          <div className="mt-5 px-3">
            <h3 className="px-2 text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1.5">
              Recent Conversations
            </h3>
            <ul className="space-y-1">
              {chatHistory.slice(0, 3).map((chat) => (
                <li key={chat.id}>
                  <button
                    onClick={() => selectChat(chat.id)}
                    className="w-full flex items-center p-2 text-left text-xs text-gray-300 rounded-md hover:bg-gray-800/50 transition-colors group"
                  >
                    <div className="flex-shrink-0 w-5 h-5 flex items-center justify-center rounded bg-gray-800 group-hover:bg-gray-700 transition-colors">
                      <MessageSquare className="h-3 w-3 text-blue-400" />
                    </div>
                    <span className="ml-2 truncate flex-grow">
                      {chat.title}
                    </span>
                    <span className="ml-1 text-[9px] text-gray-500">
                      {chat.messages}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </nav>

      {/* Document Library - when sidebar is expanded */}
      {!isSidebarCollapsed && (
        <div className="px-3 mb-4">
          <h3 className="px-2 text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1.5">
            Document Library
          </h3>
          <div className="bg-gray-800/50 rounded-lg p-2.5 shadow-inner">
            <div className="flex items-center mb-2">
              <BookOpen className="h-3.5 w-3.5 text-blue-400 mr-2" />
              <span className="text-xs font-medium">Recent Documents</span>
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center px-1.5 py-1 rounded hover:bg-gray-700/50 cursor-pointer group transition-colors">
                <FileText className="h-3 w-3 text-gray-400 group-hover:text-gray-300" />
                <span className="ml-1.5 text-[11px] text-gray-300 truncate">
                  NDA_Agreement_v2.1.pdf
                </span>
              </div>
              <div className="flex items-center px-1.5 py-1 rounded hover:bg-gray-700/50 cursor-pointer group transition-colors">
                <FileText className="h-3 w-3 text-gray-400 group-hover:text-gray-300" />
                <span className="ml-1.5 text-[11px] text-gray-300 truncate">
                  Service_Contract_2025.pdf
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* User profile section */}
      <div className="border-t border-gray-700/50 p-3">
        <div className="flex items-center">
          <div className="relative">
            <div className="bg-gradient-to-r from-blue-700 to-blue-600 p-2 rounded-full shadow-inner shadow-white/10">
              <User className="h-4 w-4 text-white" />
            </div>
            <div className="absolute bottom-0 right-0 h-2 w-2 bg-green-500 rounded-full border border-gray-800"></div>
          </div>
          {!isSidebarCollapsed && (
            <div className="ml-2.5 overflow-hidden">
              <p className="text-xs font-medium text-white truncate">
                Sarah Johnson, Esq.
              </p>
              <p className="text-[10px] text-gray-400 truncate">
                legal@johnson-associates.com
              </p>
            </div>
          )}
          {!isSidebarCollapsed && (
            <div className="ml-auto flex space-x-1">
              <button className="text-gray-400 hover:text-white p-1 rounded-full hover:bg-gray-700/50 transition-colors">
                <Settings className="h-3 w-3" />
              </button>
              <button className="text-gray-400 hover:text-white p-1 rounded-full hover:bg-gray-700/50 transition-colors">
                <LogOut className="h-3 w-3" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
