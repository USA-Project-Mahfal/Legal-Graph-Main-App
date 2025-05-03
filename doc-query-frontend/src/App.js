import React, { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');

  const handleFileUpload = async () => {
    if (!file) return alert("Please select a file first.");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      alert(data.message);
    } catch (err) {
      alert("Upload failed.");
    }
  };

  const handleAskQuestion = async () => {
    if (!question.trim()) return alert("Please enter a question.");

    const formData = new FormData();
    formData.append("question", question);

    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResponse(data.response);
    } catch (err) {
      alert("Failed to get response.");
    }
  };

  return (
    <div style={{ padding: '30px', maxWidth: '600px', margin: 'auto' }}>
      <h2>📄 Upload a Document</h2>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={handleFileUpload} style={{ marginLeft: '10px' }}>Upload</button>

      <h2 style={{ marginTop: '40px' }}>❓ Ask a Question</h2>
      <textarea
        rows="4"
        cols="60"
        placeholder="Type your question..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />
      <br />
      <button onClick={handleAskQuestion} style={{ marginTop: '10px' }}>Submit</button>

      {response && (
        <>
          <h3 style={{ marginTop: '40px' }}> Response</h3>
          <div style={{ background: '#f4f4f4', padding: '10px' }}>{response}</div>
        </>
      )}
    </div>
  );
}

export default App;
