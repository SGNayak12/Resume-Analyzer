import React from 'react'
import { Route,Routes } from 'react-router-dom'
import Start from './pages/start.jsx'
import CandidateHome from './pages/CandidateHome.jsx'
import HrHome from './pages/HrHome.jsx'


const App = () => {
  return (
    <>
      <Routes>
        <Route path="/" element={<Start />} />
        <Route path="/candidate_Home" element={<CandidateHome/>} />
        <Route path="/hr_Home" element={<HrHome/>} />
      </Routes>
    </>
  )
}

export default App

