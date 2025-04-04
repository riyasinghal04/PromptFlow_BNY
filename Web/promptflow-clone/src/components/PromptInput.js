import React from 'react';
import './PromptInput.css';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import MicIcon from '@mui/icons-material/Mic';
import SendIcon from '@mui/icons-material/Send';

function PromptInput() {
  return (
    <div className="prompt-input-container">
      <textarea
        className="prompt-input"
        placeholder="How can we assist you in refining your prompt?"
      />
      <div className="input-actions">
        <button className="action-button">
          <FileUploadIcon />
        </button>
        <button className="action-button">
          <MicIcon />
        </button>
        <button className="action-button submit">
          <SendIcon />
        </button>
      </div>
    </div>
  );
}

export default PromptInput;
