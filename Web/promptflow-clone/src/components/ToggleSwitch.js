import React from 'react';
import './ToggleSwitch.css';
import InfoIcon from '@mui/icons-material/Info';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';

function ToggleSwitch({ label, checked, onChange, modelSelector, model, onModelChange }) {
  return (
    <div className="toggle-switch-container">
      <InfoIcon className="info-icon" fontSize="small" />
      <span className="toggle-label">{label}</span>
      <label className="switch">
        <input type="checkbox" checked={checked} onChange={onChange} />
        <span className="slider round"></span>
      </label>
      {modelSelector && (
        <div className="model-selector">
          <span className="model-name">{model}</span>
          <KeyboardArrowDownIcon fontSize="small" />
        </div>
      )}
    </div>
  );
}

export default ToggleSwitch;
