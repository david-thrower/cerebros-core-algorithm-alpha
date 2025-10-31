import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { App } from '../../../UIREFERENCE/src/App';
import { MultiStageWizard } from '../../../UIREFERENCE/src/components/MultiStageWizard';
import { PromptTraining } from '../../../UIREFERENCE/src/components/PromptTraining';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/new" element={<PromptTraining />} />
        <Route path="/assistants/:id" element={<App />} />
        <Route path="/train" element={<MultiStageWizard />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
);