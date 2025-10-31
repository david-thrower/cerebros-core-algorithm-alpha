import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Dashboard } from './components/Dashboard';
import { UploadWizard } from './components/UploadWizard';
import { PromptTraining } from './components/PromptTraining';
import { Chat } from './components/Chat';
import { CreateAssistantWizard } from './components/CreateAssistantWizard';

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/upload" element={<UploadWizard />} />
        <Route path="/new" element={<PromptTraining />} />
        <Route path="/wizard" element={<CreateAssistantWizard />} />
        <Route path="/chat/:assistantId" element={<Chat />} />
      </Routes>
    </BrowserRouter>
  );
}