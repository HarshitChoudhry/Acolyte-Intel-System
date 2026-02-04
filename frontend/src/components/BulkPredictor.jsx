import React, { useState } from 'react';

const BulkPredictor = () => {
    const [file, setFile] = useState(null);
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [filterHigh, setFilterHigh] = useState(false);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setError(null);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file) {
            setError("Please select a CSV file first.");
            return;
        }

        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('https://acolyte-backend.onrender.com/predict-bulk', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Bulk processing failed');
            }

            const data = await response.json();
            setResults(data.results);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const filteredResults = filterHigh
        ? results.filter(r => r.relocation_readiness_class === 'High')
        : results;

    const downloadTemplate = () => {
        // Create CSV template with headers and sample data
        const csvContent = `student_id,gpa,test_score,study_gap,loan_sanctioned,pof_verified,uni_tier,visa_refusal,cas_issued,session_time_min,days_to_intake
STU_001,8.5,85,0,1,1,1,0,1,25,90
STU_002,7.2,75,1,1,1,2,0,1,20,120
STU_003,6.5,65,2,0,0,3,1,0,10,150`;

        // Create blob and download
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', 'student_data_template.csv');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className="glass-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
                <div>
                    <h3>Bulk Candidate Evaluation</h3>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                        Upload a CSV file with student data for batch processing.
                    </p>
                </div>
                <a
                    href="#"
                    onClick={(e) => { e.preventDefault(); downloadTemplate(); }}
                    style={{ color: 'var(--primary)', fontSize: '0.8rem', fontWeight: 600 }}
                >
                    Download Template CSV
                </a>
            </div>

            <form onSubmit={handleSubmit} style={{ marginBottom: '2rem', padding: '2rem', border: '2px dashed var(--glass-border)', borderRadius: 'var(--radius)', textAlign: 'center' }}>
                <input
                    type="file"
                    accept=".csv"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                    id="csv-upload"
                />
                <label htmlFor="csv-upload" style={{ cursor: 'pointer', display: 'block' }}>
                    <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>📁</div>
                    <p style={{ fontWeight: 600 }}>{file ? file.name : 'Click to select or drag and drop CSV'}</p>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Max file size: 5MB</p>
                </label>

                {file && (
                    <button type="submit" className="btn btn-primary" style={{ marginTop: '1.5rem' }} disabled={loading}>
                        {loading ? 'Processing Batch...' : 'Start Batch Inference'}
                    </button>
                )}
            </form>

            {error && (
                <div style={{ color: 'var(--error)', padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', borderRadius: 'var(--radius)', marginBottom: '1.5rem' }}>
                    <strong>Error:</strong> {error}
                </div>
            )}

            {results.length > 0 && (
                <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                        <h4>Results ({filteredResults.length})</h4>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                            <span style={{ fontSize: '0.875rem' }}>Show High Probability Only</span>
                            <label className="switch">
                                <input type="checkbox" checked={filterHigh} onChange={() => setFilterHigh(!filterHigh)} />
                                <span className="slider round"></span>
                            </label>
                        </div>
                    </div>

                    <div style={{ overflowX: 'auto' }}>
                        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                            <thead>
                                <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--glass-border)', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                                    <th style={{ padding: '1rem' }}>Student ID</th>
                                    <th style={{ padding: '1rem' }}>GPA</th>
                                    <th style={{ padding: '1rem' }}>Test Score</th>
                                    <th style={{ padding: '1rem' }}>Visa Likelihood</th>
                                    <th style={{ padding: '1rem' }}>Readiness</th>
                                </tr>
                            </thead>
                            <tbody>
                                {filteredResults.map((res, i) => (
                                    <tr key={i} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', fontSize: '0.9rem' }}>
                                        <td style={{ padding: '1rem', fontWeight: 500 }}>{res.student_id}</td>
                                        <td style={{ padding: '1rem' }}>{res.gpa}</td>
                                        <td style={{ padding: '1rem' }}>{res.test_score}</td>
                                        <td style={{ padding: '1rem' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                                <div style={{ width: '60px', height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px' }}>
                                                    <div style={{
                                                        width: `${res.visa_likelihood}%`,
                                                        height: '100%',
                                                        background: res.visa_likelihood >= 70 ? 'var(--success)' : 'var(--warning)',
                                                        borderRadius: '2px'
                                                    }}></div>
                                                </div>
                                                {res.visa_likelihood}%
                                            </div>
                                        </td>
                                        <td style={{ padding: '1rem' }}>
                                            <span className={`badge badge-${res.relocation_readiness_class.toLowerCase()}`}>
                                                {res.relocation_readiness_class}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            <style>{`
        .switch {
          position: relative;
          display: inline-block;
          width: 44px;
          height: 22px;
        }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider {
          position: absolute;
          cursor: pointer;
          top: 0; left: 0; right: 0; bottom: 0;
          background-color: var(--surface-hover);
          transition: .4s;
          border-radius: 34px;
        }
        .slider:before {
          position: absolute;
          content: "";
          height: 16px; width: 16px;
          left: 3px; bottom: 3px;
          background-color: white;
          transition: .4s;
          border-radius: 50%;
        }
        input:checked + .slider { background-color: var(--primary); }
        input:checked + .slider:before { transform: translateX(22px); }
      `}</style>
        </div>
    );
};

export default BulkPredictor;
