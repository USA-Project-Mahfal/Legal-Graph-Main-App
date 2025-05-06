import { Moon, Sun, HelpCircle, Settings } from "lucide-react";

export default function Header({
  activeTab,
  setActiveTab,
  isDarkMode,
  toggleTheme,
  currentChatId,
  chatHistory,
}) {
  // Get the current chat title
  const getCurrentChatTitle = () => {
    if (currentChatId === "current") return "Untitled Chat";
    const chat = chatHistory.find((c) => c.id === currentChatId);
    return chat ? chat.title : "Chat";
  };

  // Get the title based on active tab
  const getTitle = () => {
    switch (activeTab) {
      case "chat":
        return getCurrentChatTitle();
      case "upload":
        return "Document Upload";
      case "history":
        return "Chat History";
      default:
        return "Legal Intelligence";
    }
  };

  return (
    <header
      className={`px-6 py-3 ${
        isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
      } border-b shadow-sm flex items-center justify-between`}
    >
      <h1 className="text-xl font-semibold">{getTitle()}</h1>

      <div className="flex items-center space-x-3">
        <button
          onClick={toggleTheme}
          className={`p-2 rounded-full ${
            isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-100"
          }`}
          title={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
        >
          {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
        </button>

        <button
          className={`p-2 rounded-full ${
            isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-100"
          }`}
          title="Help"
        >
          <HelpCircle size={20} />
        </button>

        <button
          className={`p-2 rounded-full ${
            isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-100"
          }`}
          title="Settings"
        >
          <Settings size={20} />
        </button>
      </div>
    </header>
  );
}
