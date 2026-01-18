import React, { useState } from 'react';
import QuizDisplay from './QuizDisplay';
import '../styles/GenerateQuizTab.css';
import config from '../config';

function GenerateQuizTab() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [quiz, setQuiz] = useState(null);
  const [cached, setCached] = useState(false);

  const handleGenerateQuiz = async (e) => {
    e.preventDefault();
    
    if (!url.trim()) {
      setError('Please enter a Wikipedia URL');
      return;
    }

    setLoading(true);
    setError('');
    setQuiz(null);

    try {
      const response = await fetch(`${config.API_BASE_URL}/generate-quiz`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate quiz');
      }

      const data = await response.json();
      setQuiz(data);
      setCached(data.cached || false);
    } catch (err) {
      setError(err.message || 'Error generating quiz. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="generate-quiz-tab">
      <div className="url-input-section">
        <form onSubmit={handleGenerateQuiz}>
          <div className="input-group">
            <input
              type="url"
              placeholder="Enter Wikipedia URL (e.g., https://en.wikipedia.org/wiki/Alan_Turing)"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              disabled={loading}
            />
            <button type="submit" disabled={loading}>
              {loading ? '⏳ Generating...' : '🚀 Generate Quiz'}
            </button>
          </div>
        </form>

        {error && <div className="error-message">❌ {error}</div>}
        {cached && <div className="info-message">💾 This quiz was cached from a previous request</div>}
      </div>

      {quiz && (
        <QuizDisplay
          title={quiz.title}
          preview={quiz.article_preview}
          quizData={quiz.quiz_data}
          relatedTopics={quiz.related_topics}
          url={quiz.url}
        />
      )}

      {!quiz && !loading && !error && (
        <div className="placeholder">
          <p>📝 Enter a Wikipedia URL above to generate a quiz</p>
          <p className="examples">
            Examples:
            <br />
            • https://en.wikipedia.org/wiki/Alan_Turing
            <br />
            • https://en.wikipedia.org/wiki/Machine_Learning
            <br />
            • https://en.wikipedia.org/wiki/Albert_Einstein
          </p>
        </div>
      )}
    </div>
  );
}

export default GenerateQuizTab;


