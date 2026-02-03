import React, { useState } from 'react';
import SinglePredictor from './components/SinglePredictor';
import BulkPredictor from './components/BulkPredictor';
import './index.css';

function App() {
  const [activeTab, setActiveTab] = useState('single');

  return (
    <div className="container">
      <header style={{ textAlign: 'center', marginBottom: '3rem', marginTop: '1rem' }}>
        <h1 style={{ fontSize: '2.5rem', background: 'linear-gradient(135deg, #fff 0%, #94a3b8 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: '0.5rem' }}>
          Acolyte Predictive Intelligence
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem' }}>
          Real-time visa likelihood and student relocation readiness scoring
        </p>
      </header>

      <nav style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginBottom: '2.5rem' }}>
        <button
          className={`btn ${activeTab === 'single' ? 'btn-primary' : 'glass-card'}`}
          onClick={() => setActiveTab('single')}
          style={{ padding: '0.75rem 2rem', background: activeTab === 'single' ? undefined : 'rgba(255,255,255,0.05)' }}
        >
          Single Candidate
        </button>
        <button
          className={`btn ${activeTab === 'bulk' ? 'btn-primary' : 'glass-card'}`}
          onClick={() => setActiveTab('bulk')}
          style={{ padding: '0.75rem 2rem', background: activeTab === 'bulk' ? undefined : 'rgba(255,255,255,0.05)' }}
        >
          Bulk Upload
        </button>
      </nav>

      <main>
        {activeTab === 'single' ? <SinglePredictor /> : <BulkPredictor />}
      </main>

      <footer style={{ marginTop: '5rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
        <p>&copy; 2026 Acolyte Predictive Systems. Powered by Two-Step Stacked RandomForest Models.</p>
      </footer>
    </div>
  );
}

export default App;
