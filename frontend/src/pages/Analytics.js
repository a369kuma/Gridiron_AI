import React from 'react';
import styled from 'styled-components';
import { usePrediction } from '../context/PredictionContext';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const AnalyticsContainer = styled.div`
  max-width: 1200px;
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

const Grid = styled.div`
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
`;

const SectionTitle = styled.h2`
  color: #ffffff;
  margin-bottom: 2rem;
  font-size: 1.5rem;
  font-weight: 600;
  letter-spacing: -0.02em;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
`;

const StatItem = styled.div`
  background: rgba(20, 20, 20, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 2rem;
  border-radius: 16px;
  text-align: center;
  backdrop-filter: blur(10px);
`;

const StatValue = styled.div`
  font-size: 2.5rem;
  font-weight: 700;
  color: #d4af37;
  margin-bottom: 0.75rem;
  letter-spacing: -0.02em;
`;

const StatLabel = styled.div`
  color: #a0a0a0;
  font-size: 1rem;
  font-weight: 500;
`;

const ChartContainer = styled.div`
  height: 300px;
  margin-top: 1rem;
`;

const NoDataMessage = styled.div`
  text-align: center;
  color: #a0a0a0;
  font-weight: 500;
  padding: 3rem;
  background: rgba(20, 20, 20, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  backdrop-filter: blur(20px);
`;

const COLORS = ['#d4af37', '#ffd700', '#b8860b', '#daa520', '#cd853f'];

const Analytics = () => {
  const { predictions, modelStats } = usePrediction();

  // Calculate analytics data
  const getAnalyticsData = () => {
    if (!predictions || predictions.length === 0) {
      return {
        homeWins: 0,
        awayWins: 0,
        avgConfidence: 0,
        confidenceDistribution: [],
        teamPredictions: {}
      };
    }

    const homeWins = predictions.filter(p => p.predicted_winner === 'home').length;
    const awayWins = predictions.filter(p => p.predicted_winner === 'away').length;
    const avgConfidence = predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length;

    // Confidence distribution
    const confidenceRanges = [
      { range: '0-20%', count: 0, min: 0, max: 0.2 },
      { range: '20-40%', count: 0, min: 0.2, max: 0.4 },
      { range: '40-60%', count: 0, min: 0.4, max: 0.6 },
      { range: '60-80%', count: 0, min: 0.6, max: 0.8 },
      { range: '80-100%', count: 0, min: 0.8, max: 1.0 }
    ];

    predictions.forEach(prediction => {
      const confidence = prediction.confidence;
      const range = confidenceRanges.find(r => confidence >= r.min && confidence < r.max);
      if (range) {
        range.count++;
      }
    });

    // Team predictions
    const teamPredictions = {};
    predictions.forEach(prediction => {
      const homeTeam = prediction.home_team;
      const awayTeam = prediction.away_team;
      
      if (!teamPredictions[homeTeam]) teamPredictions[homeTeam] = { wins: 0, total: 0 };
      if (!teamPredictions[awayTeam]) teamPredictions[awayTeam] = { wins: 0, total: 0 };
      
      teamPredictions[homeTeam].total++;
      teamPredictions[awayTeam].total++;
      
      if (prediction.predicted_winner === 'home') {
        teamPredictions[homeTeam].wins++;
      } else {
        teamPredictions[awayTeam].wins++;
      }
    });

    const teamData = Object.entries(teamPredictions)
      .map(([team, stats]) => ({
        team,
        winRate: stats.total > 0 ? (stats.wins / stats.total) * 100 : 0,
        total: stats.total
      }))
      .sort((a, b) => b.winRate - a.winRate)
      .slice(0, 10);

    return {
      homeWins,
      awayWins,
      avgConfidence,
      confidenceDistribution: confidenceRanges,
      teamData
    };
  };

  const analyticsData = getAnalyticsData();

  if (!predictions || predictions.length === 0) {
    return (
      <AnalyticsContainer>
        <Title>Analytics Dashboard</Title>
        <NoDataMessage>
          No prediction data available yet. Make some predictions to see analytics!
        </NoDataMessage>
      </AnalyticsContainer>
    );
  }

  const pieData = [
    { name: 'Home Wins', value: analyticsData.homeWins },
    { name: 'Away Wins', value: analyticsData.awayWins }
  ];

  return (
    <AnalyticsContainer>
      <Title>Analytics Dashboard</Title>
      
      <StatsGrid>
        <StatItem>
          <StatValue>{predictions.length}</StatValue>
          <StatLabel>Total Predictions</StatLabel>
        </StatItem>
        
        <StatItem>
          <StatValue>{(analyticsData.avgConfidence * 100).toFixed(1)}%</StatValue>
          <StatLabel>Avg Confidence</StatLabel>
        </StatItem>
        
        <StatItem>
          <StatValue>{analyticsData.homeWins}</StatValue>
          <StatLabel>Home Predictions</StatLabel>
        </StatItem>
        
        <StatItem>
          <StatValue>{analyticsData.awayWins}</StatValue>
          <StatLabel>Away Predictions</StatLabel>
        </StatItem>
        
        {modelStats && modelStats.accuracy !== null && (
          <StatItem>
            <StatValue>{(modelStats.accuracy * 100).toFixed(1)}%</StatValue>
            <StatLabel>Model Accuracy</StatLabel>
          </StatItem>
        )}
      </StatsGrid>

      <Grid>
        <Card>
          <SectionTitle>Home vs Away Predictions</SectionTitle>
          <ChartContainer>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Card>

        <Card>
          <SectionTitle>Confidence Distribution</SectionTitle>
          <ChartContainer>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={analyticsData.confidenceDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#667eea" />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Card>

        <Card style={{ gridColumn: '1 / -1' }}>
          <SectionTitle>Top Teams by Predicted Win Rate</SectionTitle>
          <ChartContainer>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={analyticsData.teamData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 100]} />
                <YAxis dataKey="team" type="category" width={60} />
                <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, 'Win Rate']} />
                <Bar dataKey="winRate" fill="#48bb78" />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Card>
      </Grid>
    </AnalyticsContainer>
  );
};

export default Analytics;
