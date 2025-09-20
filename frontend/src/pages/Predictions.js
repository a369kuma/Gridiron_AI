import React from 'react';
import styled from 'styled-components';
import { usePrediction } from '../context/PredictionContext';
import RecentPredictions from '../components/RecentPredictions';

const PredictionsContainer = styled.div`
  max-width: 1000px;
  margin: 0 auto;
`;

const Title = styled.h1`
  color: #ffffff;
  font-size: 3rem;
  font-weight: 700;
  text-align: center;
  margin-bottom: 2rem;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #ffffff 0%, #d4af37 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
`;

const Card = styled.div`
  background: rgba(20, 20, 20, 0.8);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 2.5rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
`;

const SectionTitle = styled.h2`
  color: #ffffff;
  margin-bottom: 2rem;
  font-size: 1.5rem;
  font-weight: 600;
  letter-spacing: -0.02em;
`;

const Predictions = () => {
  const { predictions } = usePrediction();

  return (
    <PredictionsContainer>
      <Title>Recent Predictions</Title>
      
      <Card>
        <SectionTitle>All Predictions</SectionTitle>
        <RecentPredictions />
      </Card>
    </PredictionsContainer>
  );
};

export default Predictions;
