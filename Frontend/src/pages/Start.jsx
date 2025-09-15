import React from 'react'
import {Link} from "react-router-dom"
import Image from "../assets/Resume_Pic.jpg"
const Start = () => {
  return (
    <>
        <div className='bg-cover bg-center h-screen' style={{backgroundImage:`url(${Image})`}}>
          <div className='flex items-center justify-center'>
            <div className=''>
            <Link to='/candidate_login'>As a Candidate</Link>
            <Link to='/hr_login'>As a HR</Link>
            </div>
          </div>
        </div>
    </>
  )
}

export default Start
