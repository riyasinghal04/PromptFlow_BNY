import React from 'react';
import './Header.css';

function Header() {
  return (
    <header className="header">
      <div className="logo">
        <img src="/promptflow-logo.png" alt="PromptFlow" className="logo-img" />
        <span className="logo-text">PromptFlow</span>
      </div>
      <div className="share-button">
        <button>Share</button>
      </div>
    </header>
  );
}

export default Header;
