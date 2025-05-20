import { useState, useEffect, useRef } from "react";
import MessageList from "./chat/MessageList";
import ChatInput from "./chat/ChatInput";
import DocumentViewer from "./DocumentViewer";

export default function ChatView({ messages, setMessages, isDarkMode }) {
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [category, setCategory] = useState("skip");
  const [selectedDocument, setSelectedDocument] = useState(null);
  const messagesEndRef = useRef(null);

  const categories = [
    'License_Agreements',
    'Maintenance', 
    'Service',
    'Sponsorship',
    'Strategic Alliance',
    'skip'
  ];

  // Handle message submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage = { id: Date.now(), role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: input,
          category: category
        }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      
      const systemResponse = {
        id: Date.now() + 1,
        role: "system",
        content: data.response,
        highlighted_document: data.highlighted_document
      };
      
      setMessages((prev) => [...prev, systemResponse]);
    } catch (error) {
      console.error("Error:", error);
      const errorResponse = {
        id: Date.now() + 1,
        role: "system",
        content: "Sorry, there was an error processing your request."
      };
      setMessages((prev) => [...prev, errorResponse]);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle key press for chat input
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex flex-col w-full h-full">
      <div className={`p-3 ${isDarkMode ? "bg-gray-800" : "bg-gray-50"}`}>
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className={`w-full p-2 rounded-lg border ${
            isDarkMode 
              ? "bg-gray-700 text-white border-gray-600" 
              : "bg-white text-gray-800 border-gray-300"
          }`}
        >
          {categories.map((cat) => (
            <option key={cat} value={cat}>{cat}</option>
          ))}
        </select>
      </div>

      <MessageList
        messages={messages}
        isLoading={isLoading}
        messagesEndRef={messagesEndRef}
        isDarkMode={isDarkMode}
        onViewDocument={setSelectedDocument}
      />

      <ChatInput
        input={input}
        setInput={setInput}
        handleSubmit={handleSubmit}
        handleKeyDown={handleKeyDown}
        isDarkMode={isDarkMode}
      />

      {selectedDocument && (
        <DocumentViewer
          document={selectedDocument}
          onClose={() => setSelectedDocument(null)}
          isDarkMode={isDarkMode}
        />
      )}
    </div>
  );
}
