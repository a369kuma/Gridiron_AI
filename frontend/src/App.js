import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import styled, { createGlobalStyle } from 'styled-components';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import Predictions from './pages/Predictions';
import Analytics from './pages/Analytics';
import { PredictionProvider } from './context/PredictionContext';

const GlobalStyle = createGlobalStyle`
  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
      'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
      sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    background: #0a0a0a;
    min-height: 100vh;
    color: #ffffff;
    line-height: 1.6;
  }

  code {
    font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', 'Courier New',
      monospace;
  }

  h1, h2, h3, h4, h5, h6 {
    font-weight: 600;
    letter-spacing: -0.025em;
  }
`;

const AppContainer = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: #0a0a0a;
`;

const MainContent = styled.main`
  flex: 1;
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
  background: linear-gradient(180deg, #0a0a0a 0%, #111111 100%);
`;

function App() {
  return (
    <PredictionProvider>
      <Router>
        <GlobalStyle />
        <AppContainer>
          <Header />
          <MainContent>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/analytics" element={<Analytics />} />
            </Routes>
          </MainContent>
        </AppContainer>
      </Router>
    </PredictionProvider>
  );
}

export default App;
