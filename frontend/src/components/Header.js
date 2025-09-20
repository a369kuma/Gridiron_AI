import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import styled from 'styled-components';

const HeaderContainer = styled.header`
  background: rgba(10, 10, 10, 0.95);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  padding: 1.5rem 0;
  position: sticky;
  top: 0;
  z-index: 100;
`;

const HeaderContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Logo = styled(Link)`
  font-size: 1.5rem;
  font-weight: 700;
  color: #ffffff;
  text-decoration: none;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  letter-spacing: -0.02em;
  
  &:hover {
    color: #d4af37;
  }
`;

const LogoIcon = styled.div`
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, #d4af37 0%, #ffd700 100%);
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 900;
  color: #000;
  font-size: 0.9rem;
`;

const Nav = styled.nav`
  display: flex;
  gap: 2rem;
`;

const NavLink = styled(Link)`
  color: #a0a0a0;
  text-decoration: none;
  font-weight: 500;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  transition: all 0.3s ease;
  position: relative;
  font-size: 0.95rem;
  letter-spacing: -0.01em;
  
  &:hover {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.05);
  }
  
  ${props => props.$active && `
    color: #d4af37;
    background: rgba(212, 175, 55, 0.1);
    
    &::after {
      content: '';
      position: absolute;
      bottom: -8px;
      left: 50%;
      transform: translateX(-50%);
      width: 4px;
      height: 4px;
      background: #d4af37;
      border-radius: 50%;
    }
  `}
`;

const Header = () => {
  const location = useLocation();
  
  return (
    <HeaderContainer>
      <HeaderContent>
        <Logo to="/">
          <LogoIcon>NFL</LogoIcon>
          Gridiron AI
        </Logo>
        <Nav>
          <NavLink to="/" $active={location.pathname === '/'}>
            Dashboard
          </NavLink>
          <NavLink to="/predictions" $active={location.pathname === '/predictions'}>
            Predictions
          </NavLink>
          <NavLink to="/analytics" $active={location.pathname === '/analytics'}>
            Analytics
          </NavLink>
        </Nav>
      </HeaderContent>
    </HeaderContainer>
  );
};

export default Header;
