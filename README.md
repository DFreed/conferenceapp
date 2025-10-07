# Conference Attendee Prioritizer

A powerful Streamlit application that intelligently prioritizes conference attendees using local LLM integration via Ollama. Perfect for sales teams, business development, and conference organizers who need to identify the most relevant contacts from large attendee lists.

## 🚀 Features

### Core Functionality
- **CSV Upload & Processing** - Upload conference attendee lists in any CSV format
- **Intelligent Name Parsing** - Automatically splits full names into first/last components
- **Smart Role Classification** - Uses both rule-based logic and local LLM for accurate role identification
- **Query-Based Ranking** - Natural language queries to find specific types of contacts
- **Export Results** - Download prioritized contact lists as CSV

### Advanced Features
- **Local LLM Integration** - Uses Ollama for privacy-preserving AI classification
- **Performance Optimized** - Rule-based pre-filtering + batch LLM processing + caching
- **Real-time Progress Tracking** - Visual progress bars and detailed processing statistics
- **Flexible Model Selection** - Support for multiple Ollama models (1.5B to 8B parameters)
- **Persistent Caching** - Instant repeat runs with disk-based role classification cache

## 🎯 Use Cases

- **Sales Prospecting** - Find decision makers and technical contacts at conferences
- **Business Development** - Identify potential partners and customers
- **Conference Planning** - Prioritize networking opportunities
- **Lead Qualification** - Filter out sales/consulting roles to focus on buyers
- **Technical Outreach** - Target engineers, procurement, and operations personnel

## 📋 Requirements

- Python 3.8+
- Ollama (for local LLM functionality)
- At least 4GB RAM (8GB+ recommended for larger models)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/DFreed/conferenceapp.git
cd conferenceapp
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install and Setup Ollama

#### Windows
```bash
# Install via winget
winget install Ollama.Ollama

# Or download from https://ollama.ai
```

#### macOS
```bash
# Install via Homebrew
brew install ollama

# Or download from https://ollama.ai
```

#### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 5. Download Models
```bash
# Start Ollama service
ollama serve

# Download recommended models (in separate terminal)
ollama pull qwen2.5:1.5b-instruct  # Fastest, smallest
ollama pull qwen2.5:3b-instruct    # Balanced
ollama pull llama3.1:8b            # Highest quality
```

## 🚀 Usage

### 1. Start the Application
```bash
# Make sure Ollama is running
ollama serve

# Start Streamlit app
streamlit run app.py
```

### 2. Upload Your Data
- Upload a CSV file with conference attendee information
- Supported columns: `name`, `first_name`, `last_name`, `company`, `title`, `email`, `linkedin_url`
- The app will automatically parse full names if you have a `name` column

### 3. Configure LLM Settings
- **Enable/Disable LLM**: Toggle local AI classification
- **Select Model**: Choose from available Ollama models
- **Batch Size**: Adjust processing speed (smaller = faster)

### 4. Enter Your Query
Use natural language to describe what you're looking for:
- "high voltage transmission engineers"
- "utility procurement managers in the Southeast"
- "grid-scale storage decision makers"
- "substation operations personnel"

### 5. Review and Export
- View ranked results with role classifications
- Download prioritized CSV for your CRM or outreach tools

## 📊 Performance

### Processing Speed
- **Rule-based classification**: ~1000 titles/second
- **LLM classification**: ~25-100 titles/second (depending on model)
- **Cached results**: Instant (sub-second)

### Model Performance
| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| qwen2.5:1.5b-instruct | 986 MB | Fastest | Good | Large lists, quick filtering |
| qwen2.5:3b-instruct | 1.9 GB | Fast | Better | Balanced performance |
| llama3.1:8b | 4.9 GB | Slower | Best | High accuracy needs |

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set default model
export OLLAMA_DEFAULT_MODEL=qwen2.5:3b-instruct

# Optional: Increase Ollama timeout
export OLLAMA_TIMEOUT=30
```

### Customization
- **Role Categories**: Modify `ROLE_LABELS` in `app.py` to add custom role types
- **Filtering Rules**: Update `_RULES` regex patterns for different industries
- **Scoring Weights**: Adjust `INCLUDE_HINTS` and boost values for your use case

## 📁 Project Structure

```
conferenceapp/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── .role_cache.json      # LLM classification cache (auto-generated)
└── README.md             # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [Ollama](https://ollama.ai/) for local LLM capabilities
- [scikit-learn](https://scikit-learn.org/) for text similarity algorithms
- [pandas](https://pandas.pydata.org/) for data processing

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/DFreed/conferenceapp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DFreed/conferenceapp/discussions)

## 🔄 Changelog

### v1.0.0
- Initial release with core functionality
- Local LLM integration via Ollama
- Rule-based + AI-powered role classification
- Performance optimizations and caching
- Comprehensive name parsing
- Real-time progress tracking

---

**Made with ❤️ for conference organizers and sales teams**
