import React, { useState } from 'react';
import './App.css';
import PromptInput from './components/PromptInput';
import ToggleSwitch from './components/ToggleSwitch';
import Header from './components/Header';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

function App() {
  const [darkMode] = useState(true);
  const [inherit, setInherit] = useState(false);
  const [compress, setCompress] = useState(false);
  const [convert, setConvert] = useState(true);
  const [model, setModel] = useState('GPT-4o');

  const darkTheme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      background: {
        default: '#1a1a1a',
        paper: '#2d2d2d',
      },
      primary: {
        main: '#3f7ccc',
      },
    },
  });

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <div className="App">
        <Header />
        <main className="main-content">
          <div className="icon-container">
            <div className="prompt-icon">
              <div className="folder-icon"></div>
            </div>
          </div>
          <PromptInput />
          <div className="toggle-container">
            <ToggleSwitch 
              label="inherit" 
              checked={inherit} 
              onChange={() => setInherit(!inherit)} 
            />
            <ToggleSwitch 
              label="compress" 
              checked={compress} 
              onChange={() => setCompress(!compress)} 
            />
            <ToggleSwitch 
              label="convert" 
              checked={convert} 
              onChange={() => setConvert(!convert)} 
              modelSelector
              model={model}
              onModelChange={setModel}
            />
          </div>
        </main>
      </div>
    </ThemeProvider>
  );
}

export default App;

