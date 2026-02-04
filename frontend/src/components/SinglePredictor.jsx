import React, { useState } from 'react';

const SinglePredictor = () => {
  const [formData, setFormData] = useState({
    gpa: 8.5,
    test_score: 85,
    study_gap: 0,
    loan_sanctioned: 1,
    pof_verified: 1,
    uni_tier: 2,
    visa_refusal: 0,
    cas_issued: 1,
    session_time_min: 30,
    days_to_intake: 60
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'gpa' || name === 'test_score' || name === 'session_time_min'
        ? parseFloat(value)
        : parseInt(value)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('https://acolyte-backend.onrender.com/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Prediction failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-2">
      <div className="glass-card">
        <h3>Student Details</h3>
        <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
          Enter academic and engagement metrics for prediction.
        </p>
        <form onSubmit={handleSubmit}>
          <div className="grid grid-cols-2">
            <div className="input-group">
              <label>GPA (0.0 - 10.0)</label>
              <input type="number" name="gpa" step="0.1" value={formData.gpa} onChange={handleChange} required min="0" max="10" />
            </div>
            <div className="input-group">
              <label>Test Score (0 - 100)</label>
              <input type="number" name="test_score" value={formData.test_score} onChange={handleChange} required min="0" max="100" />
            </div>
          </div>

          <div className="grid grid-cols-2">
            <div className="input-group">
              <label>Study Gap (Years)</label>
              <input type="number" name="study_gap" value={formData.study_gap} onChange={handleChange} required min="0" />
            </div>
            <div className="input-group">
              <label>University Tier</label>
              <select name="uni_tier" value={formData.uni_tier} onChange={handleChange}>
                <option value="1">Tier 1 (Elite)</option>
                <option value="2">Tier 2 (Mid-range)</option>
                <option value="3">Tier 3 (Emerging)</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2">
            <div className="input-group">
              <label>Loan Sanctioned</label>
              <select name="loan_sanctioned" value={formData.loan_sanctioned} onChange={handleChange}>
                <option value="1">Yes</option>
                <option value="0">No</option>
              </select>
            </div>
            <div className="input-group">
              <label>POF Verified</label>
              <select name="pof_verified" value={formData.pof_verified} onChange={handleChange}>
                <option value="1">Yes</option>
                <option value="0">No</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2">
            <div className="input-group">
              <label>CAS Issued</label>
              <select name="cas_issued" value={formData.cas_issued} onChange={handleChange}>
                <option value="1">Yes</option>
                <option value="0">No</option>
              </select>
            </div>
            <div className="input-group">
              <label>Visa Refusal</label>
              <select name="visa_refusal" value={formData.visa_refusal} onChange={handleChange}>
                <option value="1">Yes</option>
                <option value="0">No</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2">
            <div className="input-group">
              <label>Session Time (Min)</label>
              <input type="number" name="session_time_min" value={formData.session_time_min} onChange={handleChange} required min="0" />
            </div>
            <div className="input-group">
              <label>Days to Intake</label>
              <input type="number" name="days_to_intake" value={formData.days_to_intake} onChange={handleChange} required min="0" />
            </div>
          </div>

          <button type="submit" className="btn btn-primary" style={{ width: '100%' }} disabled={loading}>
            {loading ? 'Analyzing...' : 'Generate Prediction'}
          </button>
        </form>
      </div>

      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        {error && (
          <div style={{ color: 'var(--error)', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: 'var(--radius)' }}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {!result && !error && !loading && (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
            <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📊</div>
            <h4>Ready to Analyze</h4>
            <p>Enter student details and click generate to see the predictive results.</p>
          </div>
        )}

        {loading && (
          <div style={{ textAlign: 'center' }}>
            <div className="spinner" style={{ marginBottom: '1rem' }}></div>
            <h4>Processing Data</h4>
            <p>Our models are calculating visa likelihood and relocation readiness...</p>
          </div>
        )}

        {result && (
          <div>
            <div style={{ marginBottom: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ fontWeight: 600 }}>Visa Approval Likelihood</span>
                <span className="badge badge-high" style={{
                  background: result.visa_likelihood >= 70 ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                  color: result.visa_likelihood >= 70 ? 'var(--success)' : 'var(--error)'
                }}>
                  {result.visa_likelihood}%
                </span>
              </div>
              <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px' }}>
                <div style={{
                  width: `${result.visa_likelihood}%`,
                  height: '100%',
                  background: result.visa_likelihood >= 70 ? 'var(--success)' : 'var(--warning)',
                  borderRadius: '4px',
                  transition: 'width 0.5s ease-out'
                }}></div>
              </div>
            </div>

            <div style={{ marginBottom: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ fontWeight: 600 }}>Relocation Readiness</span>
                <span className={`badge badge-${result.relocation_readiness_class.toLowerCase()}`}>
                  {result.relocation_readiness_class}
                </span>
              </div>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                Probability Score: {(result.relocation_readiness_score * 100).toFixed(2)}%
              </p>
            </div>

            <div style={{ padding: '1.5rem', background: 'rgba(99, 102, 241, 0.1)', borderLeft: '4px solid var(--primary)', borderRadius: '4px' }}>
              <h4 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>Recommendation</h4>
              <p style={{ fontSize: '0.95rem' }}>{result.recommendation}</p>
            </div>
          </div>
        )}
      </div>

      <style>{`
        .spinner {
          width: 40px;
          height: 40px;
          border: 4px solid rgba(255, 255, 255, 0.1);
          border-left-color: var(--primary);
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin: 0 auto;
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default SinglePredictor;
