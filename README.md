# Paper2Code - AI-Powered Scientific Paper to Code Implementation Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Agno Framework](https://img.shields.io/badge/Framework-Agno-green.svg)](https://docs.agno.com/)

## About

**Paper2Code** is an intelligent AI agent system designed to bridge the gap between academic research and practical implementation. It automatically transforms scientific papers into production-ready, well-documented code repositories. By leveraging advanced multi-agent architecture and intelligent caching strategies, Paper2Code accelerates the journey from research to deployment.

This project addresses a critical need in the research community: making groundbreaking academic work accessible and implementable for developers and researchers. Instead of manually deciphering complex mathematical formulations and algorithms from papers, Paper2Code automates the entire process.

## Key Features

### Core Capabilities
- **Multi-format Paper Processing**: Accept PDF files, arXiv URLs, and DOI identifiers
- **Intelligent Multi-Agent System**: Specialized subagents for paper analysis, code generation, and quality assurance
- **GitHub Integration**: Automated repository creation with complete project structure
- **Code Quality Assurance**: Comprehensive evaluation framework ensuring research fidelity
- **Production-Ready Output**: Generate well-structured, documented, and tested code

### Technical Highlights
- **Multi-level Caching**: Intelligent cache warming for optimal performance
- **Modular Architecture**: Configurable output levels and flexible service selection
- **Multi-Language Support**: Python (primary) with JavaScript/TypeScript support
- **Agno Framework Integration**: Leverages advanced multi-agent capabilities

## Quick Start

### Prerequisites
- Python 3.8 or higher
- GitHub Personal Access Token
- OpenAI API Key (or compatible LLM provider)
- Redis server (for caching)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ash-Blanc/paper2code.git
cd paper2code

# Install dependencies
uv sync --dev

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Basic Usage

```python
from app.main import Paper2CodeAgent
from app.models.paper import PaperMetadata, Author
from app.models.repository import RepositoryConfig, Visibility

# Initialize the agent
agent = Paper2CodeAgent()

# Create paper metadata
paper_metadata = PaperMetadata(
    title="Attention Is All You Need",
    authors=[Author(name="Ashish Vaswani"), Author(name="Noam Shazeer")],
    journal="NeurIPS",
    publication_year=2017,
    doi="10.48550/arXiv.1706.03762",
    url="https://arxiv.org/abs/1706.03762",
    domain="Natural Language Processing",
)

# Create repository configuration
repo_config = RepositoryConfig(
    description="Code implementation for Attention Is All You Need",
    visibility=Visibility.PUBLIC,
)

# Process the paper
result = agent.process_paper(
    paper_input="https://arxiv.org/abs/1706.03762",
    input_type="arxiv_url",
    repository_config=repo_config
)

print(f"Repository created: {result.repository_url}")
```

### Command Line Interface

```bash
# Process a paper from arXiv
uv run paper2code -- --input https://arxiv.org/abs/1706.03762 --type arxiv_url

# Process a paper from PDF
uv run paper2code -- --input paper.pdf --type pdf

# Process a paper from DOI
uv run paper2code -- --input 10.48550/arXiv.1706.03762 --type doi
```

## Architecture

### Specialized Subagents

1. **Paper Analysis Agent** - Extracts algorithms, mathematical formulations, and experimental setups
2. **Research Agent** - Finds similar papers on GitHub and analyzes existing implementations
3. **Architecture Agent** - Designs project structure and selects appropriate technology stack
4. **Code Generation Agent** - Generates clean, efficient code with Python as primary language
5. **Documentation Agent** - Creates comprehensive README and inline documentation
6. **Quality Assurance Agent** - Validates code correctness and performance metrics

### Intelligent Caching Strategy

The system implements multi-level caching:
- **Paper Analysis Cache** (30-day TTL): Parsed paper metadata
- **Research Results Cache** (7-day TTL): GitHub search results
- **Code Templates Cache** (90-day TTL): Common algorithm implementations
- **Integration Config Cache** (60-day TTL): Service configurations

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/unit/
uv run pytest tests/integrations/

# Run with coverage report
uv run pytest --cov=app --cov-report=html
```

## Configuration

### Environment Variables

```bash
# GitHub Configuration
GITHUB_TOKEN=your_github_personal_access_token
GITHUB_ORGANIZATION=paper2code-repos

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.7

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/paper2code
REDIS_URL=redis://localhost:6379

# Application Configuration
DEBUG=True
LOG_LEVEL=INFO
CACHE_TTL=2592000  # 30 days in seconds
```

## Project Structure

```
paper2code/
├── app/
│   ├── agents/          # Specialized agent implementations
│   ├── cache/           # Caching system and warming strategies
│   ├── integrations/    # External service integrations
│   ├── models/          # Data models and schemas
│   ├── tools/           # Utility functions
│   └── main.py          # Main agent entry point
├── tests/
│   ├── unit/            # Unit tests
│   ├── integrations/    # Integration tests
│   └── scenarios/       # End-to-end scenario tests
├── prompts/             # LLM prompt templates
├── docs/                # Documentation
├── examples/            # Usage examples
├── pyproject.toml       # Project configuration
└── README.md            # This file
```

## Development

### Running in Development Mode

```bash
# Run with hot reload
uv run uvicorn app.main:app --reload

# Run with debug mode
uv run paper2code -- --debug
```

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install dev dependencies (`uv sync --dev`)
4. Make your changes and add tests
5. Run tests (`uv run pytest`)
6. Submit a pull request

**Code Style Requirements:**
- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings for public functions
- Include comprehensive error handling
- Write tests for new functionality

## Roadmap

### Phase 1 (Current)
- ✅ Multi-agent architecture implementation
- ✅ GitHub integration
- ✅ Caching system
- ✅ Evaluation framework
- ✅ Basic documentation

### Phase 2 (Next)
- [ ] Advanced caching strategies
- [ ] Multi-language support expansion
- [ ] CI/CD integration
- [ ] Performance optimization

### Phase 3 (Future)
- [ ] Docker containerization
- [ ] Cloud deployment templates
- [ ] Advanced analytics dashboard
- [ ] REST API for third-party integration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Agno Framework](https://docs.agno.com/) - The underlying multi-agent framework
- [LangWatch](https://langwatch.ai/) - Prompt management and monitoring
- [GitHub](https://github.com/) - Repository hosting and integration

## Support

- **Issues**: [GitHub Issues](https://github.com/Ash-Blanc/paper2code/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Ash-Blanc/paper2code/discussions)
- **Documentation**: [Full Documentation](https://paper2code.readthedocs.io/)

---

**Paper2Code** - Making scientific research accessible through code 🚀
