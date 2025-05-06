import { useState } from 'react';
import Header from './components/Header';
import ChatView from './components/ChatView';
import UploadView from './components/UploadView';
import HistoryView from './components/HistoryView';
import GraphView from './components/GraphView';
import Footer from './components/Footer';
import Sidebar from './components/Sidebar';

export default function App() {
  const [activeTab, setActiveTab] = useState('chat'); // 'chat', 'upload', 'history'
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'system',
      content:
        "Welcome to Legal Intelligence. You can ask questions about your legal documents, and I'll provide answers based on the content you've uploaded.",
    },
  ]);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState({});
  const [chatHistory, setChatHistory] = useState([
    {
      id: 'chat-1',
      title: 'NDA Review Discussion',
      date: '2 May 2025',
      messages: 4,
    },
    {
      id: 'chat-2',
      title: 'Contract Interpretation',
      date: '30 Apr 2025',
      messages: 7,
    },
    {
      id: 'chat-3',
      title: 'Legal Compliance Inquiry',
      date: '28 Apr 2025',
      messages: 3,
    },
  ]);
  const [currentChatId, setCurrentChatId] = useState('current');

  // Handle file upload
  const handleFileUpload = (files) => {
    const newFiles = Array.from(files).map((file) => {
      const fileId = `upload-${Date.now()}-${file.name}`;
      setUploadProgress((prev) => ({ ...prev, [fileId]: 0 }));

      // Simulate upload progress
      const interval = setInterval(() => {
        setUploadProgress((prev) => {
          const newProgress = Math.min((prev[fileId] || 0) + 10, 100);
          if (newProgress === 100) clearInterval(interval);
          return { ...prev, [fileId]: newProgress };
        });
      }, 300);

      return { id: fileId, name: file.name, size: file.size, type: file.type };
    });

    setUploadedFiles((prev) => [...prev, ...newFiles]);
  };

  // Remove file
  const removeFile = (id) => {
    setUploadedFiles((prev) => prev.filter((file) => file.id !== id));
    setUploadProgress((prev) => {
      const newProgress = { ...prev };
      delete newProgress[id];
      return newProgress;
    });
  };

  // Select a chat from history
  const selectChat = (chatId) => {
    setCurrentChatId(chatId);
    setActiveTab('chat');

    // For this demo, simulate different messages based on the chat ID
    if (chatId === 'chat-1') {
      setMessages([
        {
          id: 1,
          role: 'system',
          content: 'Welcome to the NDA Review Discussion.',
        },
        {
          id: 2,
          role: 'user',
          content: 'Can we share data with third parties under NDA X?',
        },
        {
          id: 3,
          role: 'system',
          content:
            'Based on Section 4.2 of NDA X, data sharing with third parties is permitted only with prior written consent and when the third party signs a comparable confidentiality agreement.',
          sources: [
            { title: 'NDA X', section: 'Section 4.2', relevance: 0.92 },
          ],
        },
      ]);
    } else if (chatId === 'chat-2') {
      setMessages([
        {
          id: 1,
          role: 'system',
          content: 'Welcome to Contract Interpretation.',
        },
        {
          id: 2,
          role: 'user',
          content:
            'What are our obligations under section 7 of the service contract?',
        },
      ]);
    } else {
      // Reset to default for new chat
      setMessages([
        {
          id: 1,
          role: 'system',
          content:
            "Welcome to Legal Intelligence. You can ask questions about your legal documents, and I'll provide answers based on the content you've uploaded.",
        },
      ]);
    }
  };

  // Create new chat
  const createNewChat = () => {
    setCurrentChatId('current');
    setMessages([
      {
        id: 1,
        role: 'system',
        content:
          "Welcome to Legal Intelligence. You can ask questions about your legal documents, and I'll provide answers based on the content you've uploaded.",
      },
    ]);
    setActiveTab('chat');
  };

  // Toggle theme
  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div
      className={`flex h-screen ${
        isDarkMode ? 'bg-gray-800 text-white' : 'bg-gray-50 text-gray-900'
      }`}
    >
      {/* Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        createNewChat={createNewChat}
        chatHistory={chatHistory}
        selectChat={selectChat}
        isSidebarCollapsed={isSidebarCollapsed}
        setIsSidebarCollapsed={setIsSidebarCollapsed}
      />

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          isDarkMode={isDarkMode}
          toggleTheme={toggleTheme}
          currentChatId={currentChatId}
          chatHistory={chatHistory}
        />

        <main
          className={`flex-grow flex overflow-hidden ${
            isDarkMode ? 'bg-gray-900' : 'bg-white'
          }`}
        >
          {activeTab === 'chat' && (
            <ChatView
              messages={messages}
              setMessages={setMessages}
              isDarkMode={isDarkMode}
              currentChatId={currentChatId}
            />
          )}

          {activeTab === 'upload' && (
            <UploadView
              uploadedFiles={uploadedFiles}
              uploadProgress={uploadProgress}
              handleFileUpload={handleFileUpload}
              removeFile={removeFile}
              isDarkMode={isDarkMode}
            />
          )}

          {activeTab === 'graph' && <GraphView />}

          {activeTab === 'history' && (
            <HistoryView
              chatHistory={chatHistory}
              selectChat={selectChat}
              createNewChat={createNewChat}
              isDarkMode={isDarkMode}
            />
          )}
        </main>

        <Footer isDarkMode={isDarkMode} />
      </div>
    </div>
  );
}
