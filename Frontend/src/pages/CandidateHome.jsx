import React from "react";

const CandidateHome = () => {
  return (
    <div className="h-screen flex flex-col">
      <nav className="w-full bg-gray-900 text-white py-4 px-8 flex justify-between items-center shadow-md">
        <h2 className="text-xl font-bold">Candidate Dashboard</h2>
        <div className="flex gap-6">
          <a href="#profile" className="hover:text-gray-300 transition">
            Profile
          </a>
        </div>
      </nav>

      <div className="flex flex-1 items-center justify-center">
        <div className="w-[90%] md:w-[50%] bg-white p-8 rounded-lg shadow-lg text-center">
          <h1 className="text-3xl font-bold mb-6">Upload Your Resume</h1>
          <div className="flex flex-col items-center gap-4">
            <input
              className="border rounded-md p-3 w-full"
              type="file"
              name="resume"
              id="file"
              multiple
            />
            <button className="px-6 py-3 bg-green-600 text-white text-xl font-semibold rounded-lg shadow-lg hover:bg-green-700 transition">
              Submit Resume
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CandidateHome;
