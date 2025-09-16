import React from "react";
import { Link } from "react-router-dom";
import Image from "../assets/Resume_Pic.jpg";

const Start = () => {
  return (
    <div
      className="bg-cover bg-center h-screen relative"
      style={{ backgroundImage: `url(${Image})` }}
    >
      {/* Overlay to improve text visibility */}
      {/* <div className="absolute inset-0 bg-black bg-opacity-50"></div> */}

      {/* Main content */}
      <div className="relative z-10 flex flex-col items-center justify-center h-full text-center text-black px-4">
        {/* Heading */}
        <h1 className="text-5xl font-bold mb-4 drop-shadow-lg">
          Resume Analyzer
        </h1>
        <p className="text-lg mb-10 max-w-xl drop-shadow-md">
          Upload, analyze, and match resumes with job opportunities using AI-powered insights.
        </p>

        {/* Buttons */}
        <div className="flex flex-col md:flex-row gap-6">
          <Link
            to="/candidate_Home"
            className="px-6 py-3 bg-white text-black text-xl font-semibold rounded-lg shadow-lg hover:bg-gray-200 transition"
          >
            As a Candidate
          </Link>
          <Link
            to="/hr_Home"
            className="px-6 py-3 bg-white text-black text-xl font-semibold rounded-lg shadow-lg hover:bg-gray-200 transition"
          >
            As a HR
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Start;
