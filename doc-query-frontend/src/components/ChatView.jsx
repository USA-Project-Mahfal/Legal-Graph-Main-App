import { useState, useEffect, useRef } from "react";
import MessageList from "./chat/MessageList";
import ChatInput from "./chat/ChatInput";

export default function ChatView({ messages, setMessages, isDarkMode }) {
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Handle message submission
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage = { id: Date.now(), role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Simulate API response delay
    setTimeout(() => {
      // Add system response with source documents
      const systemResponse = {
        id: Date.now() + 1,
        role: "system",
        content: generateSampleResponse(input),
        sources: [
          {
            title: "NDA Agreement v2.1",
            section: "Section 4.2 - Data Sharing",
            relevance: 0.89,
          },
          {
            title: "Third Party Data Policy",
            section: "Compliance Requirements",
            relevance: 0.75,
          },
        ],
      };
      setMessages((prev) => [...prev, systemResponse]);
      setIsLoading(false);
    }, 2000);
  };

  // Sample response generator
  const generateSampleResponse = (query) => {
    const responses = [
      "Based on the retrieved documents, specifically Section 4.2 of the NDA Agreement v2.1, data sharing with third parties is permitted under the following conditions: (1) the third party must sign a comparable NDA with confidentiality terms at least as restrictive as NDA X, (2) written approval must be obtained prior to sharing, and (3) a record of all shared data must be maintained. The Third Party Data Policy further specifies that any such sharing requires a Data Processing Addendum when personal information is involved.",
      "According to the legal documents analyzed, the statute of limitations for filing this type of claim is 3 years from the date of discovery. However, the agreement specifically mentions a reduced period of 2 years for contractual disputes in Section 7.3.",
      "The contract does not explicitly address this scenario. While Clause 12.4 covers force majeure events, remote work arrangements due to public health emergencies are not specifically included. I recommend seeking clarification through a formal amendment to the agreement.",
    ];

    return responses[Math.floor(Math.random() * responses.length)];
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
      <MessageList
        messages={messages}
        isLoading={isLoading}
        messagesEndRef={messagesEndRef}
        isDarkMode={isDarkMode}
      />

      <ChatInput
        input={input}
        setInput={setInput}
        handleSubmit={handleSubmit}
        handleKeyDown={handleKeyDown}
        isDarkMode={isDarkMode}
      />
    </div>
  );
}
