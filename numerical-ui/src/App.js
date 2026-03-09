// ✅ App.js (Router Shell Only)
// ✅ Purpose: App.js should ONLY define routes.
// ✅ Your actual UI rendering (the big JSX you already have) belongs in Home.js.
// ✅ This enables shareable URLs like /run/:runId without touching your current rendering structure.

import React from "react"; // ✅ no hooks here; App is just a wrapper
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"; 
// ✅ BrowserRouter: enables URL-based navigation
// ✅ Routes/Route: declare routes
// ✅ Navigate: redirect unknown routes to "/"
import ExperimentsDashboard from "./ExperimentsDashboard";
import Home from "./Home"; 
// ✅ Home must contain your existing UI (compare + charts + tables)
// ✅ IMPORTANT: Home.js should export default function Home() { ... }

import RunViewer from "./RunViewer"; 
// ✅ RunViewer shows a persisted "shared run" loaded from backend (/runs/{id})
// ✅ It is mounted at /run/:runId

export default function App() {
  return (
    // ✅ Wrap entire app with BrowserRouter so URL routes work
    <BrowserRouter>
      <Routes>
        {/* ✅ Main page: your existing UI lives here */}
        <Route path="/" element={<Home />} />

        {/* ✅ Shared run page: used for "Share Run" links */}
        <Route path="/run/:runId" element={<RunViewer />} />

        <Route path="/experiments" element={<ExperimentsDashboard />} />

        {/* ✅ Catch-all: if user types random URL, send them back home */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}