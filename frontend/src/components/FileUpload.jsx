import React, { useState } from 'react';

function FileUpload({ onDocumentUploaded, onError }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      onError('Please select a file first.');
      return;
    }

    setIsUploading(true);
    onError(''); // Clear previous errors

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/api/v1/document/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        onDocumentUploaded(data.session_id, selectedFile.name, data.message);
      } else {
        const errorData = await response.json();
        onError(errorData.detail || 'File upload failed.');
      }
    } catch (error) {
      console.error('Upload error:', error);
      onError('Network error during file upload. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="file-upload-container">
      <h2>Upload Document</h2>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={isUploading || !selectedFile}>
        {isUploading ? 'Uploading...' : 'Upload & Start Chat'}
      </button>
    </div>
  );
}

export default FileUpload;