"""
LinkedIn Job Search Module

This module provides functionality for searching and retrieving job postings
from LinkedIn using the unofficial LinkedIn API. It includes job type mapping,
job search, and detailed job information retrieval.

WARNING: This uses an unofficial LinkedIn API which may violate LinkedIn's 
Terms of Service. Use at your own risk.

Dependencies:
    - linkedin-api: Unofficial LinkedIn API wrapper
    - nest-asyncio: For handling async operations in Jupyter/Streamlit environments
"""

import os
import asyncio
from typing import List, Dict, Optional, Any
import nest_asyncio

# Apply nest_asyncio to handle event loop conflicts in Jupyter/Streamlit
nest_asyncio.apply()

from linkedin_api import Linkedin


# Initialize LinkedIn API client with credentials from environment variables
try:
    api = Linkedin(os.environ["LINKEDIN_EMAIL"], os.environ["LINKEDIN_PASS"])
    print("LinkedIn API initialized successfully")
except KeyError as e:
    print(f"Missing required environment variable: {e}")
    api = None
except Exception as e:
    print(f"Error initializing LinkedIn API: {str(e)}")
    api = None


def get_job_type(job_type: str) -> Optional[str]:
    """
    Convert human-readable job type to LinkedIn API job type code.
    
    LinkedIn uses single-letter codes to represent different job types.
    This function maps common job type descriptions to these codes.
    
    Args:
        job_type (str): Human-readable job type (e.g., "full-time", "contract")
        
    Returns:
        Optional[str]: Single-letter code for LinkedIn API, or None if not found
        
    Example:
        >>> get_job_type("full-time")
        'F'
        >>> get_job_type("internship")
        'I'
    """
    # Mapping of job types to LinkedIn API codes
    job_type_mapping = {
        "full-time": "F",
        "contract": "C", 
        "part-time": "P",
        "temporary": "T",
        "internship": "I",
        "volunteer": "V",
        "other": "O"
    }
    
    # Return the mapped code, case-insensitive lookup
    return job_type_mapping.get(job_type.lower())


def get_job_ids(
    keywords: str, 
    location_name: str, 
    job_type: Optional[str] = None,
    limit: int = 10,
    companies: Optional[List[str]] = None,
    industries: Optional[List[str]] = None,
    remote: Optional[str] = None
) -> List[str]:
    """
    Search for job postings on LinkedIn and return job IDs.
    
    This function performs a search on LinkedIn using specified criteria
    and returns a list of job IDs that can be used to fetch detailed information.
    
    Args:
        keywords (str): Search keywords for job titles/descriptions
        location_name (str): Geographic location for job search
        job_type (Optional[str]): Type of job (full-time, contract, etc.)
        limit (int): Maximum number of jobs to return (default: 10)
        companies (Optional[List[str]]): List of company names to filter by
        industries (Optional[List[str]]): List of industries to filter by
        remote (Optional[str]): Remote work preference
        
    Returns:
        List[str]: List of job IDs for detailed retrieval
        
    Raises:
        Exception: If LinkedIn API is not initialized or search fails
    """
    if api is None:
        raise Exception("LinkedIn API not initialized. Check credentials.")
    
    # Convert job type to LinkedIn API format
    if job_type is not None:
        job_type = get_job_type(job_type)
    
    print(f"Searching jobs with parameters:")
    print(f"  Keywords: {keywords}")
    print(f"  Location: {location_name}")
    print(f"  Job Type: {job_type}")
    print(f"  Limit: {limit}")
    print(f"  Companies: {companies}")
    print(f"  Industries: {industries}")
    print(f"  Remote: {remote}")

    try:
        # Perform job search using LinkedIn API
        job_postings = api.search_jobs(
            keywords=keywords,
            job_type=job_type,
            location_name=location_name,
            companies=companies,
            industries=industries,
            remote=remote,
            limit=limit
        )
        
        # Extract job IDs from tracking URNs
        # LinkedIn returns tracking URNs in format "jobPosting:XXXXXXXX"
        job_ids = [
            job['trackingUrn'].split('jobPosting:')[1] 
            for job in job_postings 
            if 'trackingUrn' in job and 'jobPosting:' in job['trackingUrn']
        ]
        
        print(f"Found {len(job_ids)} job postings")
        return job_ids
        
    except Exception as e:
        print(f"Error searching for jobs: {str(e)}")
        return []


async def get_job_details(job_id: str) -> Dict[str, Any]:
    """
    Retrieve detailed information for a specific job posting.
    
    This async function fetches comprehensive job details from LinkedIn
    including company information, job description, location, and application URL.
    
    Args:
        job_id (str): LinkedIn job ID to fetch details for
        
    Returns:
        Dict[str, Any]: Dictionary containing job details with the following keys:
            - company_name: Name of the hiring company
            - company_url: Company's LinkedIn URL
            - job_desc_text: Full job description text
            - work_remote_allowed: Whether remote work is allowed
            - job_title: Title of the job position
            - company_apply_url: Direct application URL
            - job_location: Job location information
    """
    if api is None:
        return _get_empty_job_dict()
    
    try:
        # Fetch job data from LinkedIn API
        job_data = api.get_job(job_id)
        
        # Safely extract nested data with fallbacks
        company_details = job_data.get('companyDetails', {})
        company_info = company_details.get(
            'com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany', {}
        )
        company_resolution = company_info.get('companyResolutionResult', {})
        
        apply_method = job_data.get('applyMethod', {})
        offsite_apply = apply_method.get('com.linkedin.voyager.jobs.OffsiteApply', {})
        
        # Construct job data dictionary with safe defaults
        job_data_dict = {
            "company_name": company_resolution.get('name', 'N/A'),
            "company_url": company_resolution.get('url', 'N/A'),
            "job_desc_text": job_data.get('description', {}).get('text', 'N/A'),
            "work_remote_allowed": job_data.get('workRemoteAllowed', False),
            "job_title": job_data.get('title', 'N/A'),
            "company_apply_url": offsite_apply.get('companyApplyUrl', 'N/A'),
            "job_location": job_data.get('formattedLocation', 'N/A')
        }
        
        print(f"Successfully fetched details for job: {job_data_dict.get('job_title', 'Unknown')}")
        return job_data_dict
        
    except Exception as e:
        print(f"Error fetching job details for job ID {job_id}: {str(e)}")
        return _get_empty_job_dict()


def _get_empty_job_dict() -> Dict[str, str]:
    """
    Return an empty job dictionary structure for error cases.
    
    Returns:
        Dict[str, str]: Empty job data structure with default values
    """
    return {
        "company_name": 'N/A',
        "company_url": 'N/A',
        "job_desc_text": 'N/A',
        "work_remote_allowed": False,
        "job_title": 'N/A',
        "company_apply_url": 'N/A',
        "job_location": 'N/A'
    }


async def fetch_all_jobs(job_ids: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch detailed information for multiple job postings.
    
    This function processes a list of job IDs and retrieves detailed information
    for each one. It can process jobs in batches to manage API rate limits.
    
    Args:
        job_ids (List[str]): List of LinkedIn job IDs
        batch_size (int): Number of jobs to process in each batch (currently not used)
        
    Returns:
        List[Dict[str, Any]]: List of job detail dictionaries
        
    Note:
        Currently processes jobs sequentially. Batch processing can be implemented
        for better performance and rate limit management.
    """
    results = []
    total_jobs = len(job_ids)
    
    print(f"Fetching details for {total_jobs} jobs...")
    
    for index, job_id in enumerate(job_ids, 1):
        print(f"Processing job {index}/{total_jobs}: {job_id}")
        
        job_detail = await get_job_details(job_id)
        results.append(job_detail)
        
        # Optional: Add delay between requests to respect rate limits
        # await asyncio.sleep(0.5)
    
    print(f"Completed fetching details for {len(results)} jobs")
    return results


async def job_threads(job_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Main entry point for fetching multiple job details asynchronously.
    
    This function serves as the primary interface for retrieving detailed
    information about multiple job postings.
    
    Args:
        job_ids (List[str]): List of LinkedIn job IDs to process
        
    Returns:
        List[Dict[str, Any]]: List of detailed job information dictionaries
    """
    return await fetch_all_jobs(job_ids, batch_size=10)


# Utility functions for enhanced functionality

def validate_search_parameters(**kwargs) -> Dict[str, Any]:
    """
    Validate search parameters before making API calls.
    
    Args:
        **kwargs: Search parameters to validate
        
    Returns:
        Dict[str, Any]: Validation results and cleaned parameters
    """
    errors = []
    
    # Check required parameters
    if not kwargs.get('keywords'):
        errors.append("Keywords are required")
    
    if not kwargs.get('location_name'):
        errors.append("Location name is required")
    
    # Validate limit
    limit = kwargs.get('limit', 10)
    if not isinstance(limit, int) or limit < 1 or limit > 100:
        errors.append("Limit must be an integer between 1 and 100")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'cleaned_params': kwargs
    }


def format_job_summary(job_details: Dict[str, Any]) -> str:
    """
    Format job details into a readable summary string.
    
    Args:
        job_details (Dict[str, Any]): Job details dictionary
        
    Returns:
        str: Formatted job summary
    """
    return f"""
Job Title: {job_details.get('job_title', 'N/A')}
Company: {job_details.get('company_name', 'N/A')}
Location: {job_details.get('job_location', 'N/A')}
Remote Work: {'Yes' if job_details.get('work_remote_allowed') else 'No'}
Apply URL: {job_details.get('company_apply_url', 'N/A')}
---
Description: {job_details.get('job_desc_text', 'N/A')[:200]}...
"""