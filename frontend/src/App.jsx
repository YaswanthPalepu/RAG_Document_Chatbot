import React, { useState } from 'react';
import FileUpload from './components/FileUpload';
import ChatWindow from './components/ChatWindow';
import './App.css';

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [uploadMessage, setUploadMessage] = useState('');
  const [error, setError] = useState('');

  const handleDocumentUploaded = (newSessionId, uploadedFileName, message) => {
    setSessionId(newSessionId);
    setFileName(uploadedFileName);
    setUploadMessage(message);
    setError('');
    console.log(`Session ID received: ${newSessionId}`);
  };

  const handleError = (errorMessage) => {
    setError(errorMessage);
    setUploadMessage('');
  };

  const handleClearSession = async () => {
    if (sessionId) {
      try {
        // Corrected DELETE endpoint path
        const response = await fetch(`http://localhost:8000/api/v1/document/session/${sessionId}`, {
          method: 'DELETE',
        });
        if (response.ok) {
          setSessionId(null);
          setFileName(null);
          setUploadMessage('Session cleared. You can now upload a new document.');
          setError('');
          console.log('Session cleared successfully.');
        } else {
          const data = await response.json();
          setError(data.detail || 'Failed to clear session.');
        }
      } catch (err) {
        setError('Network error while clearing session.');
        console.error('Error clearing session:', err);
      }
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Document Q&A System</h1>
      </header>
      <main>
        {!sessionId ? (
          <FileUpload onDocumentUploaded={handleDocumentUploaded} onError={handleError} />
        ) : (
          <div className="document-info">
            <p>Document: <strong>{fileName}</strong> uploaded. Ready to chat!</p>
            <button onClick={handleClearSession} className="clear-button">Clear Document & Session</button>
          </div>
        )}
        {uploadMessage && <p className="success-message">{uploadMessage}</p>}
        {error && <p className="error-message">{error}</p>}
        {sessionId && (
          <ChatWindow sessionId={sessionId} /> 
        )}
      </main>
    </div>
  );
}

export default App;