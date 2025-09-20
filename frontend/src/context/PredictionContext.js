import React, { createContext, useContext, useReducer, useEffect } from 'react';
import axios from 'axios';

const PredictionContext = createContext();

const initialState = {
  teams: [],
  predictions: [],
  modelStats: null,
  loading: false,
  error: null
};

const predictionReducer = (state, action) => {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload, loading: false };
    case 'SET_TEAMS':
      return { ...state, teams: action.payload };
    case 'SET_PREDICTIONS':
      return { ...state, predictions: action.payload };
    case 'ADD_PREDICTION':
      return { ...state, predictions: [action.payload, ...state.predictions] };
    case 'SET_MODEL_STATS':
      return { ...state, modelStats: action.payload };
    case 'CLEAR_ERROR':
      return { ...state, error: null };
    default:
      return state;
  }
};

export const PredictionProvider = ({ children }) => {
  const [state, dispatch] = useReducer(predictionReducer, initialState);

  // API base URL
  const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5003/api';

  // Load teams on component mount
  useEffect(() => {
    loadTeams();
    loadModelStats();
    loadRecentPredictions();
  }, []);

  const loadTeams = async () => {
    try {
      const response = await axios.get(`${API_BASE}/teams`);
      console.log('Teams loaded:', response.data.teams);
      console.log('Number of teams:', response.data.teams.length);
      dispatch({ type: 'SET_TEAMS', payload: response.data.teams });
    } catch (error) {
      console.error('Error loading teams:', error);
      // Fallback teams list if API is not available
      const fallbackTeams = [
        'NE', 'KC', 'BUF', 'MIA', 'NYJ', 'PIT', 'BAL', 'CLE', 'CIN', 'HOU',
        'IND', 'TEN', 'JAX', 'DEN', 'LV', 'LAC', 'DAL', 'PHI', 'WAS', 'NYG',
        'GB', 'MIN', 'CHI', 'DET', 'TB', 'NO', 'ATL', 'CAR', 'SF', 'SEA', 'LAR', 'ARI'
      ];
      console.log('Using fallback teams:', fallbackTeams);
      dispatch({ type: 'SET_TEAMS', payload: fallbackTeams });
      dispatch({ type: 'SET_ERROR', payload: 'API unavailable, using fallback teams' });
    }
  };

  const loadRecentPredictions = async (limit = 10) => {
    try {
      const response = await axios.get(`${API_BASE}/predictions?limit=${limit}`);
      dispatch({ type: 'SET_PREDICTIONS', payload: response.data.predictions });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: 'Failed to load predictions' });
    }
  };

  const loadModelStats = async () => {
    try {
      const response = await axios.get(`${API_BASE}/model/accuracy`);
      dispatch({ type: 'SET_MODEL_STATS', payload: response.data });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: 'Failed to load model statistics' });
    }
  };

  const getTeamStats = async (team) => {
    try {
      const response = await axios.get(`${API_BASE}/team/${team}/stats`);
      return response.data.stats;
    } catch (error) {
      console.error('Error getting team stats:', error);
      // Return default stats if API fails
      return {
        win_pct: 0.5,
        off_rating: 100.0,
        def_rating: 100.0,
        avg_points: 25.0,
        avg_yards: 350.0,
        turnover_diff: 0.0,
        injuries: 3,
        momentum: 0.0
      };
    }
  };

  const makePrediction = async (predictionData) => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'CLEAR_ERROR' });

    try {
      const response = await axios.post(`${API_BASE}/predict`, predictionData);
      
      if (response.data.success) {
        dispatch({ type: 'ADD_PREDICTION', payload: response.data.prediction });
        loadRecentPredictions(); // Refresh the list
        return response.data;
      } else {
        throw new Error(response.data.error || 'Prediction failed');
      }
    } catch (error) {
      const errorMessage = error.response?.data?.error || error.message || 'Failed to make prediction';
      dispatch({ type: 'SET_ERROR', payload: errorMessage });
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  const retrainModel = async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'CLEAR_ERROR' });

    try {
      const response = await axios.post(`${API_BASE}/model/retrain`);
      
      if (response.data.success) {
        loadModelStats(); // Refresh model statistics
        return response.data;
      } else {
        throw new Error(response.data.error || 'Model retraining failed');
      }
    } catch (error) {
      const errorMessage = error.response?.data?.error || error.message || 'Failed to retrain model';
      dispatch({ type: 'SET_ERROR', payload: errorMessage });
      throw error;
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  const updatePredictionResult = async (predictionId, actualWinner) => {
    try {
      const response = await axios.put(`${API_BASE}/predictions/${predictionId}`, {
        actual_winner: actualWinner
      });
      
      if (response.data.success) {
        loadRecentPredictions(); // Refresh predictions
        loadModelStats(); // Refresh model stats
        return response.data;
      } else {
        throw new Error(response.data.error || 'Failed to update prediction');
      }
    } catch (error) {
      const errorMessage = error.response?.data?.error || error.message || 'Failed to update prediction';
      dispatch({ type: 'SET_ERROR', payload: errorMessage });
      throw error;
    }
  };

  const clearError = () => {
    dispatch({ type: 'CLEAR_ERROR' });
  };

  const value = {
    ...state,
    loadTeams,
    loadRecentPredictions,
    loadModelStats,
    getTeamStats,
    makePrediction,
    retrainModel,
    updatePredictionResult,
    clearError
  };

  return (
    <PredictionContext.Provider value={value}>
      {children}
    </PredictionContext.Provider>
  );
};

export const usePrediction = () => {
  const context = useContext(PredictionContext);
  if (!context) {
    throw new Error('usePrediction must be used within a PredictionProvider');
  }
  return context;
};
