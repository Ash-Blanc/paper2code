# Development Guide

This guide provides comprehensive information for developers working on the Paper2Code agent system.

## 🏗️ Architecture Overview

### Core Components

```
paper2code/
├── app/
│   ├── agents/           # Specialized agents
│   ├── cache/            # Caching system
│   ├── integrations/     # External integrations
│   ├── models/           # Data models
│   ├── tools/            # Utility tools
│   └── main.py          # Main application
├── tests/               # Test files
├── prompts/             # LangWatch prompts
└── docs/                # Documentation
```

### Agent Architecture

The system uses a multi-agent architecture with the following components:

1. **Paper2CodeAgent** (Team): Main orchestrator
2. **PaperAnalysisAgent**: Extracts paper metadata and algorithms
3. **ResearchAgent**: Finds similar papers and implementations
4. **ArchitectureAgent**: Designs project structure
5. **CodeGenerationAgent**: Generates code implementations
6. **DocumentationAgent**: Creates documentation
7. **QualityAssuranceAgent**: Validates code quality

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- Git
- Docker (optional)
- Redis server
- PostgreSQL database

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/paper2code/paper2code.git
   cd paper2code
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   python -m app.db.init
   ```

6. **Run development server**
   ```bash
   python -m app.main --debug
   ```

### Docker Development

```bash
# Build development image
docker build -f Dockerfile.dev -t paper2code:dev .

# Run development container
docker run -it --rm \
  -v $(pwd):/app \
  -v $(pwd)/.env:/app/.env \
  -p 8000:8000 \
  paper2code:dev
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/               # Unit tests
├── integrations/       # Integration tests
├── scenarios/          # End-to-end tests
├── evaluations/        # Evaluation tests
└── fixtures/           # Test data
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html --cov-report=term

# Run specific test categories
pytest tests/unit/
pytest tests/integrations/
pytest tests/scenarios/

# Run specific test file
pytest tests/integrations/test_github_integration.py

# Run with verbose output
pytest -v

# Run with stop on first failure
pytest -x
```

### Writing Tests

#### Unit Tests

```python
import pytest
from app.models.paper import PaperMetadata, Author

def test_paper_metadata_creation():
    """Test PaperMetadata creation"""
    metadata = PaperMetadata(
        title="Test Paper",
        authors=[Author(name="Test Author", email="test@example.com")],
        journal="Test Journal",
        publication_year=2023
    )
    
    assert metadata.title == "Test Paper"
    assert len(metadata.authors) == 1
    assert metadata.publication_year == 2023
```

#### Integration Tests

```python
import pytest
from unittest.mock import Mock, patch
from app.integrations import GitHubIntegration

@pytest.fixture
def github_integration():
    """GitHub integration fixture"""
    with patch('app.integrations.github.Github') as mock_github:
        mock_github_instance = Mock()
        mock_user = Mock()
        mock_user.login = "testuser"
        mock_github_instance.get_user.return_value = mock_user
        mock_github.return_value = mock_github_instance
        
        yield GitHubIntegration(token="test_token", organization="test-org")

def test_github_integration_initialization(github_integration):
    """Test GitHub integration initialization"""
    assert github_integration.token == "test_token"
    assert github_integration.organization == "test-org"
```

#### Scenario Tests

```python
from scenario import Scenario, Given, When, Then

def test_paper_processing_scenario():
    """Test paper processing end-to-end"""
    scenario = Scenario(
        name="Paper Processing",
        description="Test end-to-end paper processing",
        steps=[
            Given("Paper2Code agent is initialized"),
            When("User submits paper URL", paper_url="https://arxiv.org/abs/1706.03762"),
            Then("Repository should be created", repository_url="https://github.com/testuser/repo")
        ]
    )
    
    scenario.run()
```

## 📊 Code Quality

### Linting and Formatting

```bash
# Run linting
flake8 app/
black app/
isort app/

# Run all formatting
black app/
isort app/
flake8 app/
```

### Type Checking

```bash
# Run type checking
mypy app/
```

### Static Analysis

```bash
# Run static analysis
bandit -r app/
safety check
```

## 🔧 Configuration

### Environment Variables

```env
# GitHub Configuration
GITHUB_TOKEN=your_github_token
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
CACHE_TTL=2592000
```

### Configuration Files

```python
# app/config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    github_token: str
    github_organization: str = "paper2code-repos"
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.7
    database_url: str
    redis_url: str = "redis://localhost:6379"
    debug: bool = False
    log_level: str = "INFO"
    cache_ttl: int = 2592000
    
    class Config:
        env_file = ".env"

settings = Settings()
```

## 🚀 Deployment

### Development Deployment

```bash
# Run with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run with gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Production Deployment

```bash
# Build production image
docker build -t paper2code:latest .

# Run production container
docker run -d \
  --name paper2code \
  -e GITHUB_TOKEN=your_token \
  -e OPENAI_API_KEY=your_key \
  -e DATABASE_URL=your_db_url \
  -e REDIS_URL=your_redis_url \
  -p 8000:8000 \
  paper2code:latest
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: paper2code
spec:
  replicas: 3
  selector:
    matchLabels:
      app: paper2code
  template:
    metadata:
      labels:
        app: paper2code
    spec:
      containers:
      - name: paper2code
        image: paper2code:latest
        ports:
        - containerPort: 8000
        env:
        - name: GITHUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: paper2code-secrets
              key: github-token
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: paper2code-secrets
              key: openai-api-key
```

## 📈 Monitoring

### Application Monitoring

```python
# app/monitoring.py
import logging
from prometheus_client import Counter, Histogram, Gauge

# Prometheus metrics
REQUEST_COUNT = Counter('paper2code_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('paper2code_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('paper2code_active_connections', 'Active connections')

def monitor_request(func):
    """Decorator for monitoring requests"""
    def wrapper(*args, **kwargs):
        REQUEST_COUNT.inc()
        with REQUEST_DURATION.time():
            return func(*args, **kwargs)
    return wrapper
```

### Logging

```python
# app/logging.py
import logging
import sys
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path("logs/paper2code.log"))
        ]
    )
```

## 🔒 Security

### Input Validation

```python
# app/validation.py
from pydantic import BaseModel, validator
from typing import Optional

class PaperInput(BaseModel):
    input: str
    input_type: str = "arxiv_url"
    
    @validator('input_type')
    def validate_input_type(cls, v):
        allowed_types = ['pdf', 'arxiv_url', 'doi']
        if v not in allowed_types:
            raise ValueError(f"input_type must be one of {allowed_types}")
        return v
    
    @validator('input')
    def validate_input(cls, v, values):
        input_type = values.get('input_type')
        if input_type == 'arxiv_url' and not v.startswith('https://arxiv.org/abs/'):
            raise ValueError("arxiv_url must start with https://arxiv.org/abs/")
        return v
```

### Authentication

```python
# app/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    token = credentials.credentials
    # Implement JWT validation logic
    return {"user_id": "123", "username": "testuser"}
```

## 🔄 CI/CD

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        pytest --cov=app --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
```

### Docker Build

```yaml
# .github/workflows/docker.yml
name: Docker Build

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to DockerHub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: paper2code/paper2code:latest
```

## 📚 Documentation

### API Documentation

```python
# app/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Paper2Code API", version="1.0.0")

class PaperRequest(BaseModel):
    input: str
    input_type: str = "arxiv_url"
    repository_config: dict = {}

@app.post("/process")
async def process_paper(request: PaperRequest):
    """Process a scientific paper"""
    try:
        result = await agent.process_paper(
            paper_input=request.input,
            input_type=request.input_type,
            repository_config=request.repository_config
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Code Documentation

```python
# app/agents/paper_analysis_agent.py
"""
Paper Analysis Agent

This agent is responsible for extracting and structuring information from scientific papers.
It supports multiple input formats and provides structured output for downstream processing.
"""

from typing import Dict, Any
from app.agents import SingleAgent
from app.models import Context
from app.models.paper import PaperMetadata

class PaperAnalysisAgent(SingleAgent):
    """
    Agent for analyzing scientific papers and extracting structured information.
    
    This agent processes papers in various formats (PDF, arXiv, DOI) and extracts:
    - Paper metadata (title, authors, publication info)
    - Algorithms and mathematical formulations
    - Experimental setups and datasets
    - Research questions and objectives
    """
    
    def __init__(self):
        super().__init__(
            name="paper_analysis",
            description="Analyze scientific papers and extract structured information",
            model="gpt-4o",
            temperature=0.3
        )
    
    def run(self, context: Context) -> Context:
        """
        Analyze paper and extract structured information.
        
        Args:
            context: Context containing paper input and metadata
            
        Returns:
            Context with enhanced paper analysis data
        """
        # Implementation details
        pass
```

## 🤝 Contributing

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Make changes**: Implement your feature
4. **Add tests**: Write comprehensive tests
5. **Run tests**: Ensure all tests pass
6. **Format code**: Use black, isort, flake8
7. **Commit changes**: `git commit -m "Add new feature"`
8. **Push branch**: `git push origin feature/new-feature`
9. **Create pull request**: Link to relevant issues

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Security considerations are addressed
- [ ] Performance impact is considered
- [ ] Error handling is comprehensive
- [ ] Logging is appropriate

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Manual testing completed

## Related Issues
Closes #123
Fixes #456

## Screenshots (if applicable)
![Screenshot](url)
```

## 🚀 Performance Optimization

### Caching Strategy

```python
# app/cache/optimization.py
from functools import lru_cache
import hashlib

def cache_key_generator(*args, **kwargs):
    """Generate consistent cache keys"""
    key_string = f"{args}-{kwargs}"
    return hashlib.sha256(key_string.encode()).hexdigest()

@lru_cache(maxsize=1000)
def expensive_operation(*args, **kwargs):
    """Cache expensive operations"""
    # Implementation
    pass
```

### Database Optimization

```python
# app/database/optimization.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

def create_optimized_database_url():
    """Create optimized database URL"""
    return "postgresql://user:password@localhost/paper2code?pool_size=20&max_overflow=30"

# Connection pooling
engine = create_engine(
    create_optimized_database_url(),
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

### Memory Optimization

```python
# app/memory/optimization.py
import gc
import psutil

def optimize_memory_usage():
    """Optimize memory usage"""
    # Force garbage collection
    gc.collect()
    
    # Monitor memory usage
    process = psutil.Process()
    memory_info = process.memory_info()
    
    if memory_info.rss > 500 * 1024 * 1024:  # 500MB
        # Trigger memory optimization
        pass
```

## 📊 Analytics

### Usage Analytics

```python
# app/analytics/usage.py
from datetime import datetime
from typing import Dict, Any

class UsageAnalytics:
    """Track usage analytics"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_processing_time': 0,
            'popular_papers': {}
        }
    
    def track_request(self, success: bool, processing_time: float, paper_input: str):
        """Track request metrics"""
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
        else:
            self.metrics['failed_requests'] += 1
        
        # Update average processing time
        current_avg = self.metrics['average_processing_time']
        total_requests = self.metrics['total_requests']
        self.metrics['average_processing_time'] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
        
        # Track popular papers
        paper_key = self._get_paper_key(paper_input)
        self.metrics['popular_papers'][paper_key] = (
            self.metrics['popular_papers'].get(paper_key, 0) + 1
        )
```

### Performance Analytics

```python
# app/analytics/performance.py
import time
from contextlib import contextmanager

@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"{operation_name} took {duration:.2f} seconds")
```

## 🔧 Troubleshooting

### Common Issues

#### Database Connection Issues

```bash
# Check database connection
python -c "from app.database import Database; db = Database(); print(db.connection)"

# Reset database
python -m app.db.reset
```

#### GitHub Token Issues

```bash
# Test GitHub token
python -c "from app.integrations import GitHubIntegration; gi = GitHubIntegration(); print(gi.validate_token())"

# Refresh GitHub token
python -m app.auth.refresh_github_token
```

#### Cache Issues

```bash
# Clear cache
python -m app.cache.clear

# Check cache status
python -c "from app.cache import CacheManager; cm = CacheManager(); print(cm.get_stats())"
```

### Debug Mode

```python
# app/debug.py
import logging
import traceback

def setup_debug_logging():
    """Setup debug logging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def debug_exception(func):
    """Decorator for debugging exceptions"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Exception in {func.__name__}: {e}")
            logging.error(traceback.format_exc())
            raise
    return wrapper
```

## 📚 Additional Resources

### Documentation Links

- [Agno Framework Documentation](https://docs.agno.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://pydantic-doc.helpmanual.io/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Redis Documentation](https://redis.io/documentation)

### Tools and Utilities

- **IDE**: VS Code with Python extensions
- **Linting**: flake8, black, isort
- **Testing**: pytest, pytest-cov
- **Type Checking**: mypy
- **Security**: bandit, safety
- **Performance**: memory-profiler, line-profiler

### Community and Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/paper2code/paper2code/issues)
- **Discussions**: [Join community discussions](https://github.com/paper2code/paper2code/discussions)
- **Discord**: [Join our Discord server](https://discord.gg/paper2code)
- **Email**: [Contact the development team](mailto:dev@paper2code.ai)

---

This development guide provides comprehensive information for working with the Paper2Code agent system. For additional questions or support, please refer to the main documentation or reach out to the development team.