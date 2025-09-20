import React from 'react';
import styled from 'styled-components';
import { usePrediction } from '../context/PredictionContext';

const PredictionsContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const PredictionItem = styled.div`
  background: rgba(20, 20, 20, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 2rem;
  backdrop-filter: blur(20px);
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    border-color: rgba(212, 175, 55, 0.3);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  }
`;

const GameTitle = styled.h4`
  color: #ffffff;
  margin-bottom: 1rem;
  font-size: 1.2rem;
  font-weight: 600;
  letter-spacing: -0.01em;
`;

const PredictionDetails = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
`;

const DetailItem = styled.div`
  color: #a0a0a0;
  font-size: 0.95rem;
  font-weight: 500;
`;

const ConfidenceBar = styled.div`
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  height: 6px;
  margin-top: 0.5rem;
  overflow: hidden;
`;

const ConfidenceFill = styled.div`
  background: linear-gradient(90deg, #d4af37 0%, #ffd700 100%);
  height: 100%;
  width: ${props => props.confidence * 100}%;
  transition: width 0.3s ease;
`;

const WinnerBadge = styled.span`
  background: linear-gradient(135deg, #d4af37 0%, #ffd700 100%);
  color: #000000;
  padding: 0.5rem 1rem;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 700;
  margin-left: 0.75rem;
`;

const NoPredictionsMessage = styled.div`
  text-align: center;
  color: #a0a0a0;
  font-weight: 500;
  padding: 2rem;
  background: rgba(20, 20, 20, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  backdrop-filter: blur(20px);
`;

const Timestamp = styled.div`
  color: #666666;
  font-size: 0.85rem;
  margin-top: 1rem;
  text-align: right;
  font-weight: 500;
`;

const RecentPredictions = () => {
  const { predictions } = usePrediction();

  if (!predictions || predictions.length === 0) {
    return (
      <NoPredictionsMessage>
        No predictions available yet. Make your first prediction above!
      </NoPredictionsMessage>
    );
  }

  return (
    <PredictionsContainer>
      {predictions.map((prediction, index) => (
        <PredictionItem key={index}>
          <GameTitle>
            {prediction.home_team} vs {prediction.away_team}
            <WinnerBadge>
              {prediction.predicted_winner}
            </WinnerBadge>
          </GameTitle>
          
          <PredictionDetails>
            <DetailItem>
              <strong>Home Win:</strong> {(prediction.home_win_probability * 100).toFixed(1)}%
            </DetailItem>
            <DetailItem>
              <strong>Away Win:</strong> {(prediction.away_win_probability * 100).toFixed(1)}%
            </DetailItem>
            <DetailItem>
              <strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(1)}%
              <ConfidenceBar>
                <ConfidenceFill confidence={prediction.confidence} />
              </ConfidenceBar>
            </DetailItem>
          </PredictionDetails>
          
          <Timestamp>
            {new Date(prediction.created_at).toLocaleString()}
          </Timestamp>
        </PredictionItem>
      ))}
    </PredictionsContainer>
  );
};

export default RecentPredictions;
