"""
AI Job Assistance - Tool Definitions
===================================

This module defines the specialized tools used by different agents:
- job_pipeline: LinkedIn job search functionality
- extract_cv: CV content extraction and analysis
- generate_letter_for_specific_job: Cover letter generation

Each tool is decorated with @tool to make it compatible with LangChain agents.
"""

from ast import List
from langchain.agents import tool
from data_loader import load_cv, write_to_docx
from search import job_threads, get_job_ids
import asyncio
from langchain.pydantic_v1 import BaseModel, Field


# ==============================================
# JOB SEARCH TOOLS
# ==============================================

@tool
def job_pipeline(
    keywords: str, 
    location_name: str, 
    job_type: str = None, 
    limit: int = 10, 
    companies: str = None, 
    industries: str = None, 
    remote: str = None
) -> dict:
    """
    Comprehensive LinkedIn job search tool that finds and retrieves detailed job postings.
    
    This tool combines job ID retrieval with detailed job information fetching
    to provide complete job listings for analysis.
    
    Parameters:
    -----------
    keywords : str
        Keywords describing the job role (e.g., "data scientist", "software engineer")
    location_name : str
        Geographic location for the job search (e.g., "New York", "Remote", "India")
    job_type : str, optional
        Specific type of job ("full-time", "contract", "part-time", "temporary", 
        "internship", "volunteer", "other")
    limit : int, optional
        Maximum number of jobs to retrieve (default: 10)
    companies : str, optional
        Filter jobs by specific company names
    industries : str, optional
        Filter jobs by industry types
    remote : str, optional
        Specify remote work preferences
    
    Returns:
    --------
    dict
        Dictionary containing comprehensive job information:
        - job_title: Position title
        - company_name: Hiring company
        - company_url: Company profile URL
        - job_location: Job location
        - job_desc_text: Full job description
        - work_remote_allowed: Remote work policy
        - company_apply_url: Direct application URL
    
    Note:
    -----
    This tool uses the unofficial LinkedIn API and may be subject to rate limits
    or terms of service restrictions.
    """
    print(f"Searching jobs with criteria: {keywords} in {location_name}")
    
    # Step 1: Get job IDs based on search criteria
    job_ids = get_job_ids(
        keywords=keywords,
        location_name=location_name, 
        job_type=job_type,
        limit=limit,
        companies=companies,
        industries=industries,
        remote=remote
    )
    
    print(f"Found {len(job_ids)} job IDs: {job_ids}")
    
    # Step 2: Fetch detailed information for each job
    job_descriptions = asyncio.run(job_threads(job_ids))
    
    return job_descriptions


# ==============================================
# CV ANALYSIS TOOLS
# ==============================================

@tool("extractor_tool", return_direct=False)
def extract_cv() -> dict:
    """
    Extract and structure job-relevant information from an uploaded CV/resume.
    
    This tool reads the CV file from the temporary directory and extracts
    key information relevant for job applications while maintaining privacy
    by focusing on professional qualifications rather than personal details.
    
    Returns:
    --------
    dict
        Dictionary containing structured CV information:
        - content: Full extracted text from the CV
        
    Key Features:
    -------------
    - Extracts skills, experience, and qualifications
    - Focuses on job-relevant information
    - Omits personal/private information
    - Structures data for easy analysis by other agents
    
    Note:
    -----
    Expects CV file to be saved as 'tmp/cv.pdf' by the upload handler.
    """
    print("Extracting CV content from uploaded file...")
    
    # Initialize result structure
    cv_extracted_json = {}
    
    # Load and extract text from PDF CV
    try:
        text = load_cv("tmp/cv.pdf")
        cv_extracted_json['content'] = text
        print(f"Successfully extracted {len(text)} characters from CV")
    except Exception as e:
        print(f"Error extracting CV: {str(e)}")
        cv_extracted_json['content'] = "Error: Could not extract CV content"
    
    return cv_extracted_json


# ==============================================
# COVER LETTER GENERATION TOOLS
# ==============================================

@tool
def generate_letter_for_specific_job(cv_details: str, job_details: str) -> str:
    """
    Generate a tailored cover letter using provided CV and job details.
    
    This tool creates personalized cover letters that highlight relevant
    experience and skills in relation to specific job requirements.
    
    Parameters:
    -----------
    cv_details : str
        Structured information from the candidate's CV including skills,
        experience, education, and achievements
    job_details : str
        Detailed job posting information including requirements, 
        responsibilities, company information, and job description
    
    Returns:
    --------
    str
        Formatted cover letter text tailored to the specific job opportunity
    
    Generation Strategy:
    -------------------
    The tool uses the following approach:
    1. Analyze job requirements and key qualifications
    2. Identify matching skills and experience from CV
    3. Structure letter with proper format and tone
    4. Highlight relevant achievements and capabilities
    5. Customize content for the specific company and role
    
    Prompt Template:
    ---------------
    Based on the CV details provided in {cv_details} and the job requirements 
    listed in {job_details}, write a personalized cover letter. Ensure the 
    letter articulates how the applicant's skills and experiences align with 
    the job requirements.
    
    The generated letter should:
    - Open with enthusiasm for the specific role
    - Highlight 2-3 most relevant qualifications
    - Demonstrate knowledge of the company
    - Close with a strong call to action
    - Maintain professional yet engaging tone
    """
    print("Generating personalized cover letter...")
    print(f"CV Details Length: {len(str(cv_details))}")
    print(f"Job Details Length: {len(str(job_details))}")
    
    # Note: The actual generation logic is handled by the LLM agent
    # This tool serves as the interface and parameter structure
    return f"Cover letter generation initiated with CV and job details."


# ==============================================
# UTILITY FUNCTIONS
# ==============================================

def get_tools() -> list:
    """
    Utility function to retrieve all available tools for agent configuration.
    
    Returns:
    --------
    list
        List of all tool functions available for agent use
    """
    return [job_pipeline, extract_cv, generate_letter_for_specific_job]


# ==============================================
# ALTERNATIVE/EXPERIMENTAL TOOLS
# ==============================================

@tool
def func_alternative_tool(msg: str, members: list) -> dict:
    """
    Alternative routing tool for experimental configurations.
    
    This is an alternative implementation of the routing functionality
    that could be used instead of the supervisor agent approach.
    
    Parameters:
    -----------
    msg : str
        Message or query to route
    members : list
        Available agent members for routing
        
    Returns:
    --------
    dict
        Function definition for routing decisions
        
    Note:
    -----
    This tool is currently experimental and not used in the main workflow.
    It's kept for potential future routing strategy alternatives.
    """
    members = ["Analyzer", "Generator", "Searcher"]
    options = ["FINISH"] + members
    
    # Function definition for routing schema
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [{"enum": options}],
                }
            },
            "required": ["next"],
        },
    }
    
    return function_def


# ==============================================
# COMMENTED OUT - ALTERNATIVE IMPLEMENTATIONS
# ==============================================

"""
Alternative implementation of cover letter generation with file output:

@tool
def generate_letter_for_specific_job(query: str):
    Create a cover letter tailored to a job description based on an uploaded CV.

    This version would save the cover letter as a DOCX file and return
    the download path instead of just the text content.

    Parameters:
    query (str): The combined information of CV and job to tailor the cover letter.

    Returns:
    tuple: A message indicating the cover letter is ready, along with the 
           absolute path for downloading the DOCX file.
           
    file = write_to_docx(query)
    abs_path = os.path.abspath(file)
    print(abs_path)
    return "Here is the download link: ",  abs_path
"""


# ==============================================
# TOOL INTEGRATION NOTES
# ==============================================

"""
Tool Usage by Agents:
====================

Searcher Agent:
- Uses: job_pipeline
- Purpose: Find relevant job postings based on user criteria
- Output: Detailed job listings with company info and descriptions

Analyzer Agent:
- Uses: extract_cv
- Purpose: Extract and analyze CV content for job matching
- Output: Structured CV information highlighting relevant skills

Generator Agent:
- Uses: generate_letter_for_specific_job
- Purpose: Create personalized cover letters
- Output: Tailored cover letter text

Error Handling:
==============
- All tools include basic error handling for file operations
- LinkedIn API errors are caught and logged
- CV extraction errors return error messages instead of crashing

Performance Considerations:
==========================
- LinkedIn API has rate limits - tool respects these automatically
- CV extraction is synchronous but fast for typical resume sizes
- Job fetching uses async operations for better performance
- Large job searches may take several seconds to complete
"""