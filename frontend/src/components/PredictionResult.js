import React from 'react';
import styled from 'styled-components';
import PredictionAnalysis from './PredictionAnalysis';

const ResultContainer = styled.div`
  background: rgba(20, 20, 20, 0.8);
  border: 1px solid rgba(212, 175, 55, 0.3);
  border-radius: 16px;
  padding: 2rem;
  margin-top: 2rem;
  backdrop-filter: blur(20px);
`;

const ResultTitle = styled.h3`
  color: #ffffff;
  margin-bottom: 1.5rem;
  font-size: 1.3rem;
  font-weight: 600;
  letter-spacing: -0.01em;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 1rem;
`;

const StatItem = styled.div`
  background: rgba(10, 10, 10, 0.8);
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

const WinnerBadge = styled.div`
  background: linear-gradient(135deg, #d4af37 0%, #ffd700 100%);
  color: #000000;
  padding: 1rem 2rem;
  border-radius: 12px;
  font-weight: 700;
  text-align: center;
  margin-top: 2rem;
  box-shadow: 0 4px 20px rgba(212, 175, 55, 0.3);
  letter-spacing: -0.01em;
`;

const ConfidenceBar = styled.div`
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  height: 8px;
  margin-top: 0.5rem;
  overflow: hidden;
`;

const ConfidenceFill = styled.div`
  background: linear-gradient(90deg, #d4af37 0%, #ffd700 100%);
  height: 100%;
  width: ${props => props.confidence * 100}%;
  transition: width 0.3s ease;
`;

const PredictionResult = ({ result }) => {
  if (!result) return null;

  const winner = result.predicted_winner === 'home' ? 'Home Team' : 'Away Team';
  const confidence = result.confidence;

  return (
    <ResultContainer>
      <ResultTitle>Prediction Result</ResultTitle>
      
      <StatsGrid>
        <StatItem>
          <StatValue>{(result.home_win_probability * 100).toFixed(1)}%</StatValue>
          <StatLabel>Home Win</StatLabel>
        </StatItem>
        
        <StatItem>
          <StatValue>{(result.away_win_probability * 100).toFixed(1)}%</StatValue>
          <StatLabel>Away Win</StatLabel>
        </StatItem>
        
        <StatItem>
          <StatValue>{(confidence * 100).toFixed(1)}%</StatValue>
          <StatLabel>Confidence</StatLabel>
          <ConfidenceBar>
            <ConfidenceFill confidence={confidence} />
          </ConfidenceBar>
        </StatItem>
      </StatsGrid>
      
      <WinnerBadge>
        Predicted Winner: {winner}
      </WinnerBadge>
      
      {result.analysis && (
        <PredictionAnalysis analysis={result.analysis} />
      )}
    </ResultContainer>
  );
};

export default PredictionResult;
