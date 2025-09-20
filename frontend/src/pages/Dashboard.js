import React, { useState } from 'react';
import styled from 'styled-components';
import { usePrediction } from '../context/PredictionContext';
import PredictionForm from '../components/PredictionForm';
import PredictionResult from '../components/PredictionResult';
import ModelStats from '../components/ModelStats';
import RecentPredictions from '../components/RecentPredictions';

const DashboardContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-bottom: 2rem;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const Card = styled.div`
  background: rgba(20, 20, 20, 0.8);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 2.5rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    border-color: rgba(212, 175, 55, 0.3);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
  }
`;

const FullWidthCard = styled(Card)`
  grid-column: 1 / -1;
`;

const Title = styled.h1`
  color: #ffffff;
  font-size: 3rem;
  font-weight: 700;
  text-align: center;
  margin-bottom: 0.5rem;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #ffffff 0%, #d4af37 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
`;

const Subtitle = styled.p`
  font-size: 1.25rem;
  color: #a0a0a0;
  text-align: center;
  margin-bottom: 4rem;
  font-weight: 400;
  letter-spacing: -0.01em;
`;

const SectionTitle = styled.h2`
  color: #ffffff;
  margin-bottom: 2rem;
  font-size: 1.5rem;
  font-weight: 600;
  letter-spacing: -0.02em;
`;

const LoadingSpinner = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
  color: #a0a0a0;
  font-size: 1.1rem;
  font-weight: 500;
`;

const ErrorMessage = styled.div`
  background: rgba(220, 38, 38, 0.1);
  color: #fca5a5;
  padding: 1.5rem;
  border-radius: 12px;
  margin: 1rem 0;
  border: 1px solid rgba(220, 38, 38, 0.2);
  font-weight: 500;
`;

const Dashboard = () => {
  const { loading, error, clearError } = usePrediction();
  const [predictionResult, setPredictionResult] = useState(null);

  const handlePredictionSuccess = (result) => {
    setPredictionResult(result);
  };

  const handleClearError = () => {
    clearError();
  };

  return (
    <>
      <Title>Gridiron AI</Title>
      <Subtitle>Advanced NFL Game Outcome Predictions</Subtitle>

      {error && (
        <ErrorMessage>
          {error}
          <button 
            onClick={handleClearError}
            style={{ 
              marginLeft: '1rem', 
              background: 'none', 
              border: 'none', 
              color: '#c53030',
              cursor: 'pointer',
              textDecoration: 'underline'
            }}
          >
            Dismiss
          </button>
        </ErrorMessage>
      )}

      <DashboardContainer>
        <Card>
          <SectionTitle>Game Prediction</SectionTitle>
          <PredictionForm onPredictionSuccess={handlePredictionSuccess} />
          {predictionResult && (
            <PredictionResult result={predictionResult} />
          )}
        </Card>

        <Card>
          <SectionTitle>Model Statistics</SectionTitle>
          {loading ? (
            <LoadingSpinner>Loading model statistics...</LoadingSpinner>
          ) : (
            <ModelStats />
          )}
        </Card>

        <FullWidthCard>
          <SectionTitle>Recent Predictions</SectionTitle>
          {loading ? (
            <LoadingSpinner>Loading recent predictions...</LoadingSpinner>
          ) : (
            <RecentPredictions />
          )}
        </FullWidthCard>
      </DashboardContainer>
    </>
  );
};

export default Dashboard;
