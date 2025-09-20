import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { usePrediction } from '../context/PredictionContext';

const Form = styled.form`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const FormGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const Label = styled.label`
  font-weight: 600;
  color: #ffffff;
  font-size: 0.95rem;
  letter-spacing: -0.01em;
`;

const Select = styled.select`
  padding: 1rem;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  font-size: 1rem;
  transition: all 0.3s ease;
  background: rgba(20, 20, 20, 0.8);
  color: #ffffff;
  font-weight: 500;

  &:focus {
    outline: none;
    border-color: #d4af37;
    box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.1);
  }

  &:disabled {
    background: rgba(20, 20, 20, 0.4);
    cursor: not-allowed;
    opacity: 0.6;
  }

  option {
    background: #1a1a1a;
    color: #ffffff;
  }
`;

const Input = styled.input`
  padding: 1rem;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  font-size: 1rem;
  transition: all 0.3s ease;
  background: rgba(20, 20, 20, 0.8);
  color: #ffffff;
  font-weight: 500;

  &:focus {
    outline: none;
    border-color: #d4af37;
    box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.1);
  }

  &:disabled {
    background: rgba(20, 20, 20, 0.4);
    cursor: not-allowed;
    opacity: 0.6;
  }

  &::placeholder {
    color: #a0a0a0;
  }
`;

const Button = styled.button`
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

const WeatherSection = styled.div`
  margin-top: 2rem;
  padding-top: 2rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
`;

const WeatherTitle = styled.h3`
  color: #ffffff;
  margin-bottom: 1.5rem;
  font-size: 1.2rem;
  font-weight: 600;
  letter-spacing: -0.01em;
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 1rem;

  @media (max-width: 480px) {
    grid-template-columns: 1fr;
  }
`;

const PredictionForm = ({ onPredictionSuccess }) => {
  const { teams, loading, makePrediction, getTeamStats } = usePrediction();
  
  // Debug: Log teams when they change
  useEffect(() => {
    console.log('Teams in PredictionForm:', teams);
    console.log('Number of teams in form:', teams.length);
  }, [teams]);
  const [formData, setFormData] = useState({
    homeTeam: '',
    awayTeam: '',
    temperature: 50,
    windSpeed: 5,
    humidity: 50
  });

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (formData.homeTeam === formData.awayTeam) {
      alert('Home and away teams must be different');
      return;
    }

    try {
      // Get team stats
      const homeStats = await getTeamStats(formData.homeTeam);
      const awayStats = await getTeamStats(formData.awayTeam);

      const predictionData = {
        home_team: formData.homeTeam,
        away_team: formData.awayTeam,
        home_stats: homeStats,
        away_stats: awayStats,
        weather: {
          temp: parseFloat(formData.temperature),
          wind: parseFloat(formData.windSpeed),
          humidity: parseFloat(formData.humidity)
        },
        rest_days: {
          home: 7,
          away: 7
        }
      };

      const result = await makePrediction(predictionData);
      onPredictionSuccess(result.prediction);
    } catch (error) {
      console.error('Prediction error:', error);
    }
  };

  return (
    <Form onSubmit={handleSubmit}>
      <FormGroup>
        <Label htmlFor="homeTeam">Home Team ({teams.length} teams available)</Label>
        <Select
          id="homeTeam"
          name="homeTeam"
          value={formData.homeTeam}
          onChange={handleInputChange}
          required
          disabled={loading}
        >
          <option value="">Select Home Team</option>
          {teams.map(team => (
            <option key={team} value={team}>{team}</option>
          ))}
        </Select>
      </FormGroup>

      <FormGroup>
        <Label htmlFor="awayTeam">Away Team ({teams.length} teams available)</Label>
        <Select
          id="awayTeam"
          name="awayTeam"
          value={formData.awayTeam}
          onChange={handleInputChange}
          required
          disabled={loading}
        >
          <option value="">Select Away Team</option>
          {teams.map(team => (
            <option key={team} value={team}>{team}</option>
          ))}
        </Select>
      </FormGroup>

      <WeatherSection>
        <WeatherTitle>Weather Conditions</WeatherTitle>
        <Grid>
          <FormGroup>
            <Label htmlFor="temperature">Temperature (°F)</Label>
            <Input
              type="number"
              id="temperature"
              name="temperature"
              value={formData.temperature}
              onChange={handleInputChange}
              min="-20"
              max="120"
              disabled={loading}
            />
          </FormGroup>

          <FormGroup>
            <Label htmlFor="windSpeed">Wind Speed (mph)</Label>
            <Input
              type="number"
              id="windSpeed"
              name="windSpeed"
              value={formData.windSpeed}
              onChange={handleInputChange}
              min="0"
              max="50"
              disabled={loading}
            />
          </FormGroup>

          <FormGroup>
            <Label htmlFor="humidity">Humidity (%)</Label>
            <Input
              type="number"
              id="humidity"
              name="humidity"
              value={formData.humidity}
              onChange={handleInputChange}
              min="0"
              max="100"
              disabled={loading}
            />
          </FormGroup>
        </Grid>
      </WeatherSection>

      <Button type="submit" disabled={loading}>
        {loading ? 'Predicting...' : 'Predict Game Outcome'}
      </Button>
    </Form>
  );
};

export default PredictionForm;
