import React from 'react'
import { Route,Routes } from 'react-router-dom'
import Start from './pages/start.jsx'
import Candidate_login from './pages/Candidate_login.jsx'
import HR_login from './pages/HR_login.jsx'


const App = () => {
  return (
    <>
      <Routes>
        <Route path="/" element={<Start />} />
        <Route path="/candidate_login" element={<Candidate_login/>} />
        <Route path="/HR_login" element={<HR_login/>} />
      </Routes>
    </>
  )
}

export default App

