import React from 'react';
import styled from 'styled-components';
import { usePrediction } from '../context/PredictionContext';

const StatsContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 1rem;
`;

const StatItem = styled.div`
  background: rgba(20, 20, 20, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 1.5rem;
  border-radius: 12px;
  text-align: center;
  backdrop-filter: blur(10px);
`;

const StatValue = styled.div`
  font-size: 1.8rem;
  font-weight: 700;
  color: #d4af37;
  margin-bottom: 0.5rem;
  letter-spacing: -0.02em;
`;

const StatLabel = styled.div`
  color: #a0a0a0;
  font-size: 0.9rem;
  font-weight: 500;
`;

const NoDataMessage = styled.div`
  text-align: center;
  color: #a0a0a0;
  font-weight: 500;
  padding: 2rem;
`;

const RetrainButton = styled.button`
  background: linear-gradient(135deg, #d4af37 0%, #ffd700 100%);
  color: #000000;
  border: none;
  padding: 1rem 2rem;
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.3s ease;
  margin-top: 2rem;
  width: 100%;
  letter-spacing: -0.01em;

  &:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(212, 175, 55, 0.4);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const ModelStats = () => {
  const { modelStats, loading, retrainModel } = usePrediction();

  const handleRetrain = async () => {
    try {
      await retrainModel();
    } catch (error) {
      console.error('Retraining failed:', error);
    }
  };

  if (!modelStats) {
    return (
      <NoDataMessage>
        No model statistics available yet
      </NoDataMessage>
    );
  }

  if (modelStats.accuracy === null) {
    return (
      <NoDataMessage>
        No predictions available yet
        <RetrainButton onClick={handleRetrain} disabled={loading}>
          {loading ? 'Retraining...' : 'Retrain Model'}
        </RetrainButton>
      </NoDataMessage>
    );
  }

  return (
    <StatsContainer>
      <StatsGrid>
        <StatItem>
          <StatValue>{(modelStats.accuracy * 100).toFixed(1)}%</StatValue>
          <StatLabel>Accuracy</StatLabel>
        </StatItem>
        
        <StatItem>
          <StatValue>{modelStats.correct_predictions}</StatValue>
          <StatLabel>Correct</StatLabel>
        </StatItem>
        
        <StatItem>
          <StatValue>{modelStats.total_predictions}</StatValue>
          <StatLabel>Total</StatLabel>
        </StatItem>
      </StatsGrid>
      
      <RetrainButton onClick={handleRetrain} disabled={loading}>
        {loading ? 'Retraining...' : 'Retrain Model'}
      </RetrainButton>
    </StatsContainer>
  );
};

export default ModelStats;
