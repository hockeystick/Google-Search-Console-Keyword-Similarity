# News Keyword Proximity Analyzer 📰

A powerful tool that combines data analysis and AI to help journalists analyze keyword relationships and generate content recommendations. This tool uses TF-IDF vectorization for keyword proximity analysis and OpenAI's GPT-4 for generating strategic content recommendations.

## Features

- **Keyword Proximity Analysis**: Calculate and visualize relationships between keywords using TF-IDF and cosine similarity
- **Interactive Heatmap**: Visual representation of keyword relationships
- **Automated Clustering**: Group related keywords based on similarity thresholds
- **AI-Powered Recommendations**: Get content strategy suggestions using OpenAI's GPT-4
- **Export Capabilities**: Download proximity matrices and analysis results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/news-keyword-analyzer.git
cd news-keyword-analyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up your OpenAI API key for AI recommendations:
   - Get one from [OpenAI's platform](https://platform.openai.com/api-keys)
   - Copy `.env.example` to `.env` and add your key:
   ```bash
   cp .env.example .env
   # Edit .env and add your API key
   ```

## Usage

This repository contains three applications for different use cases:

### Option 1: Basic Keyword Similarity Analysis
For simple keyword similarity analysis without AI features:
```bash
streamlit run proximity.py
```

### Option 2: Advanced Analysis with AI Recommendations (Recommended)
For comprehensive analysis with AI-powered content recommendations:
```bash
streamlit run Complete.py
```

**Note:** Requires an OpenAI API key. You can either:
- Enter it in the sidebar when running the app, or
- Set it as an environment variable: `export OPENAI_API_KEY='your-key-here'`, or
- Create a `.env` file (see `.env.example`)

### Option 3: Visualize Pre-computed Similarity Matrix
For visualizing an existing similarity matrix CSV:
```bash
streamlit run proximity_visualizer.py
```

## Workflow

1. Upload a CSV file containing keywords
   - File should have a column containing keywords
   - Each row should represent a unique keyword or phrase

2. Select the column containing your keywords

3. Click "Analyze Keywords" (or "Calculate Similarity") to generate:
   - Proximity matrix
   - Visualization heatmap
   - Keyword clusters
   - AI-powered content recommendations (Complete.py only)

## Input Format

Your CSV file should look something like this:

```csv
keywords
breaking news
election coverage
voter turnout
campaign finance
```

## Features in Detail

### Proximity Analysis
- Calculates similarity between all keywords using TF-IDF vectorization
- Generates a similarity matrix showing relationships between keywords
- Visualizes relationships through an interactive heatmap

### Clustering
- Groups related keywords based on similarity scores
- Adjustable threshold for cluster formation
- Visual representation of keyword clusters

### AI Recommendations
For each cluster, the tool provides:
- Content angles and story ideas
- Coverage strategy recommendations
- SEO optimization tips
- Follow-up story suggestions
- Internal linking recommendations

## Requirements

- Python 3.7+
- streamlit
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- openai
- python-dotenv

See `requirements.txt` for specific versions.

For development (testing, linting, etc.):
```bash
pip install -r requirements-dev.txt
```

## Configuration

The tool can be configured through the UI:
- Adjust clustering threshold
- Modify visualization parameters
- Select number of keywords to display in heatmap

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Powered by OpenAI's GPT-4
- Uses scikit-learn for TF-IDF vectorization
- Visualization powered by seaborn and matplotlib

## Code Structure

- `Complete.py` - Full-featured app with AI recommendations
- `proximity.py` - Basic keyword similarity analyzer
- `proximity_visualizer.py` - Visualizer for pre-computed matrices
- `utils.py` - Shared utility functions
- `config.py` - Configuration settings
- `tests/` - Unit tests

## Testing

Run tests using pytest:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=. tests/
```

## Future Enhancements

- Multiple language support
- Temporal analysis
- Trend detection
- Additional visualization options
- Enhanced export capabilities
- Consolidated single-app interface with navigation

## Need Help?

If you encounter any issues or have questions:
1. Check the existing issues on GitHub
2. Create a new issue with a detailed description of your problem
3. Include sample data if possible

## Disclaimer

This tool requires an OpenAI API key and may incur charges based on API usage. Please review OpenAI's pricing before use.
