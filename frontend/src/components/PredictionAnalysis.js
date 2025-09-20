import React from 'react';
import styled from 'styled-components';

const AnalysisContainer = styled.div`
  background: rgba(20, 20, 20, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 2rem;
  margin-top: 2rem;
  backdrop-filter: blur(20px);
`;

const AnalysisTitle = styled.h3`
  color: #ffffff;
  margin-bottom: 1.5rem;
  font-size: 1.3rem;
  font-weight: 600;
  letter-spacing: -0.01em;
`;

const ReasoningText = styled.div`
  background: rgba(10, 10, 10, 0.8);
  border: 1px solid rgba(212, 175, 55, 0.3);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 2rem;
  color: #ffffff;
  font-size: 1rem;
  line-height: 1.6;
  font-weight: 500;
`;

const FactorsGrid = styled.div`
  display: grid;
  gap: 1rem;
  margin-bottom: 2rem;
`;

const FactorItem = styled.div`
  background: rgba(10, 10, 10, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 1.5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const FactorInfo = styled.div`
  flex: 1;
`;

const FactorName = styled.div`
  color: #ffffff;
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
`;

const FactorDescription = styled.div`
  color: #a0a0a0;
  font-size: 0.9rem;
  line-height: 1.4;
`;

const ImpactBadge = styled.div`
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  
  ${props => {
    switch(props.impact) {
      case 'High':
        return `
          background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
          color: #ffffff;
        `;
      case 'Medium':
        return `
          background: linear-gradient(135deg, #d4af37 0%, #b8860b 100%);
          color: #000000;
        `;
      case 'Low':
        return `
          background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
          color: #ffffff;
        `;
      default:
        return `
          background: rgba(255, 255, 255, 0.1);
          color: #ffffff;
        `;
    }
  }}
`;

const TeamComparison = styled.div`
  background: rgba(10, 10, 10, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
`;

const ComparisonTitle = styled.h4`
  color: #ffffff;
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 1rem;
`;

const ComparisonGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
`;

const ComparisonItem = styled.div`
  text-align: center;
`;

const ComparisonLabel = styled.div`
  color: #a0a0a0;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
  font-weight: 500;
`;

const ComparisonValue = styled.div`
  color: #ffffff;
  font-size: 1.2rem;
  font-weight: 700;
  margin-bottom: 0.25rem;
`;

const ComparisonDiff = styled.div`
  color: ${props => props.positive ? '#10b981' : props.negative ? '#ef4444' : '#a0a0a0'};
  font-size: 0.8rem;
  font-weight: 600;
`;

const ConfidenceLevel = styled.div`
  background: linear-gradient(135deg, #d4af37 0%, #ffd700 100%);
  color: #000000;
  padding: 1rem 2rem;
  border-radius: 12px;
  text-align: center;
  font-weight: 700;
  font-size: 1.1rem;
  letter-spacing: -0.01em;
`;

const PredictionAnalysis = ({ analysis }) => {
  if (!analysis) return null;

  const { key_factors, team_comparison, confidence_level, prediction_reasoning } = analysis;

  return (
    <AnalysisContainer>
      <AnalysisTitle>Prediction Analysis</AnalysisTitle>
      
      <ReasoningText>
        {prediction_reasoning}
      </ReasoningText>

      <TeamComparison>
        <ComparisonTitle>Team Comparison</ComparisonTitle>
        <ComparisonGrid>
          <ComparisonItem>
            <ComparisonLabel>Offensive Rating</ComparisonLabel>
            <ComparisonValue>
              {team_comparison.offensive_rating.home} vs {team_comparison.offensive_rating.away}
            </ComparisonValue>
            <ComparisonDiff 
              positive={team_comparison.offensive_rating.difference > 0}
              negative={team_comparison.offensive_rating.difference < 0}
            >
              {team_comparison.offensive_rating.difference > 0 ? '+' : ''}{team_comparison.offensive_rating.difference.toFixed(1)}
            </ComparisonDiff>
          </ComparisonItem>
          
          <ComparisonItem>
            <ComparisonLabel>Defensive Rating</ComparisonLabel>
            <ComparisonValue>
              {team_comparison.defensive_rating.home} vs {team_comparison.defensive_rating.away}
            </ComparisonValue>
            <ComparisonDiff 
              positive={team_comparison.defensive_rating.difference > 0}
              negative={team_comparison.defensive_rating.difference < 0}
            >
              {team_comparison.defensive_rating.difference > 0 ? '+' : ''}{team_comparison.defensive_rating.difference.toFixed(1)}
            </ComparisonDiff>
          </ComparisonItem>
          
          <ComparisonItem>
            <ComparisonLabel>Win Percentage</ComparisonLabel>
            <ComparisonValue>
              {(team_comparison.win_percentage.home * 100).toFixed(1)}% vs {(team_comparison.win_percentage.away * 100).toFixed(1)}%
            </ComparisonValue>
            <ComparisonDiff 
              positive={team_comparison.win_percentage.difference > 0}
              negative={team_comparison.win_percentage.difference < 0}
            >
              {team_comparison.win_percentage.difference > 0 ? '+' : ''}{(team_comparison.win_percentage.difference * 100).toFixed(1)}%
            </ComparisonDiff>
          </ComparisonItem>
        </ComparisonGrid>
      </TeamComparison>

      {key_factors && key_factors.length > 0 && (
        <>
          <AnalysisTitle>Key Decision Factors</AnalysisTitle>
          <FactorsGrid>
            {key_factors.map((factor, index) => (
              <FactorItem key={index}>
                <FactorInfo>
                  <FactorName>{factor.factor}</FactorName>
                  <FactorDescription>{factor.description}</FactorDescription>
                </FactorInfo>
                <ImpactBadge impact={factor.impact}>
                  {factor.impact} Impact
                </ImpactBadge>
              </FactorItem>
            ))}
          </FactorsGrid>
        </>
      )}

      <ConfidenceLevel>
        Confidence Level: {confidence_level}
      </ConfidenceLevel>
    </AnalysisContainer>
  );
};

export default PredictionAnalysis;
