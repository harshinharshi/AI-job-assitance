# 🤖 AI Job Assistance Agent

A sophisticated multi-agent system built with LangGraph and LangChain that automates job searching, CV analysis, and cover letter generation. This project demonstrates how LLM-based agents can work together to streamline the job application process.

## 🎯 Features

- **Intelligent Job Search**: Automatically searches LinkedIn for relevant job opportunities
- **CV Analysis**: Extracts and analyzes skills, experience, and qualifications from uploaded resumes
- **Smart Job Matching**: Recommends the best job fits based on CV analysis
- **Automated Cover Letters**: Generates personalized cover letters for specific job applications
- **Multi-Agent Architecture**: Uses specialized agents working together through a supervisor

## 🏗️ Architecture

The system uses a multi-agent architecture with the following components:

```
                    ┌─────────────┐
                    │ SUPERVISOR  │ ← Entry point, routes tasks
                    └─────┬───────┘
           ┌──────────────┼──────────────┐
           │              │              │
    ┌──────▼──────┐ ┌────▼────┐ ┌───────▼──────┐
    │  ANALYZER   │ │SEARCHER │ │  GENERATOR   │
    │extract_cv   │ │job_pipe │ │generate_letter│
    └─────────────┘ └─────────┘ └──────────────┘
```

### Agent Responsibilities

- **👨‍💼 Supervisor**: Routes tasks between agents and determines workflow completion
- **📊 Analyzer**: Extracts and analyzes CV content using the `extract_cv` tool
- **🔍 Searcher**: Finds relevant job opportunities using the `job_pipeline` tool
- **✍️ Generator**: Creates personalized cover letters using the `generate_letter_for_specific_job` tool

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-job-assistance
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file with the following variables:
   ```env
   # Required API Keys
   OPENAI_API_KEY=<your-openai-api-key>
   LINKEDIN_EMAIL=<your-linkedin-email>
   LINKEDIN_PASS=<your-linkedin-password>
   
   # Optional: LangSmith Tracing
   LANGCHAIN_API_KEY=<your-langsmith-key>
   LANGCHAIN_TRACING_V2=true
   
   # LLM Provider Selection
   LLM_NAME=openai  # Options: openai, groq, llama3
   ```

## 💻 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload your CV** (PDF format) using the sidebar file uploader

3. **Interact with the agent** using predefined queries or custom requests:
   - "Extract and summarize my CV"
   - "Find me Data scientist job in India"
   - "Generate a cover letter for my CV"

## 🔧 Configuration

### LLM Provider Options

| Provider | Model | Stability | Notes |
|----------|-------|-----------|-------|
| OpenAI | GPT-4 | ✅ Stable | Recommended for production |
| Groq | Llama3-70B | ⚠️ Unstable | Due to routing/token limits |
| Local Llama3 | Llama3 | ⚠️ Unstable | Requires local setup |

### LinkedIn API Integration

⚠️ **Important Notice**: This project uses an unofficial LinkedIn API. Using it might violate LinkedIn's Terms of Service. Use at your own risk.

- **Source**: [linkedin-api by tomquirk](https://github.com/tomquirk/linkedin-api)
- **Risk**: Potential account restrictions or API changes

## 📁 Project Structure

```
├── agents.py           # Multi-agent system definition and graph creation
├── app.py             # Streamlit web interface (DO NOT EDIT)
├── data_loader.py     # Document processing utilities (DO NOT EDIT)
├── llms.py           # LLM provider configurations (DO NOT EDIT)
├── prompts.py        # Prompt templates for different LLM providers
├── search.py         # LinkedIn job search functionality (DO NOT EDIT)
├── tools.py          # LangChain tools for agents (DO NOT EDIT)
├── requirements.txt  # Python dependencies
├── .gitignore       # Git ignore patterns
└── README.md        # Project documentation
```

## 🛠️ Development

### Workflow Process

1. **Entry Point**: System starts at Supervisor with user input
2. **Routing Decision**: Supervisor analyzes current state and selects next agent
3. **Agent Execution**: Selected agent performs task using its specialized tool
4. **Report Back**: Agent returns results to Supervisor via shared state
5. **Continue or Finish**: Supervisor decides to route to another agent or finish

### Example Query Flow

```
User: "Find data science jobs in Germany and write a cover letter"
  ↓
Supervisor → Searcher (job search) → Supervisor → Analyzer (CV analysis) 
  ↓
Supervisor → Generator (cover letter) → Supervisor → FINISH
```

## 🐛 Known Issues & TODO

### Current Limitations
- **LinkedIn Search**: Limited search parameters, needs enrichment

### Planned Improvements
- [ ] Enhance LinkedIn search with more parameters
- [ ] Fix Groq/Llama3 stability issues
- [ ] Add more sophisticated job matching algorithms
- [ ] Implement better error handling and retry mechanisms
- [ ] Add support for multiple CV formats (DOCX, TXT)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## ⚠️ Disclaimer

This tool is for educational and research purposes. The use of unofficial APIs may violate terms of service of respective platforms. Users are responsible for ensuring compliance with applicable terms and conditions.

---

**Inspiration**: Multi-agent example from [LangGraph Documentation](https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/agent_supervisor.ipynb?ref=blog.langchain.dev)