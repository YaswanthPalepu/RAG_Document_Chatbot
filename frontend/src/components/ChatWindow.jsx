import React, { useState, useEffect, useRef } from 'react';

function ChatWindow({ sessionId }) {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const handleQueryChange = (event) => {
    setQuery(event.target.value);
  };

  const handleSubmitQuery = async (event) => {
    event.preventDefault();
    if (!query.trim()) return;

    const userMessage = { sender: 'user', text: query };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setQuery('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/v1/chat/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ session_id: sessionId, query: query }),
      });

      if (response.ok) {
        const data = await response.json();
        const botMessage = { sender: 'bot', text: data.answer, sources: data.sources };
        setMessages((prevMessages) => [...prevMessages, botMessage]);
      } else {
        const errorData = await response.json();
        const errorMessage = errorData.detail || 'Failed to get an answer.';
        setMessages((prevMessages) => [...prevMessages, { sender: 'bot', text: `Error: ${errorMessage}` }]);
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages((prevMessages) => [...prevMessages, { sender: 'bot', text: 'Network error. Please try again.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-window-container">
      <h2>Ask about your document</h2>
      <div className="messages-display">
        {messages.length === 0 && <p className="no-messages">Upload a document and start asking questions!</p>}
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            <p><strong>{msg.sender === 'user' ? 'You' : 'Bot'}:</strong> {msg.text}</p>
            {msg.sources && msg.sources.length > 0 && (
              <small className="sources">Sources: {msg.sources.join(', ')}</small>
            )}
          </div>
        ))}
        {isLoading && <div className="loading-indicator">Bot is thinking...</div>}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmitQuery} className="chat-input-form">
        <input
          type="text"
          value={query}
          onChange={handleQueryChange}
          placeholder="Type your question here..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !query.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}

export default ChatWindow;