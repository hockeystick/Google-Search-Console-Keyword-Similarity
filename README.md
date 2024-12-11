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

3. Set up your OpenAI API key. You can get one from [OpenAI's platform](https://platform.openai.com/).

## Usage

1. Start the application:
```bash
streamlit run news_analyzer.py
```

2. Enter your OpenAI API key in the sidebar

3. Upload a CSV file containing keywords
   - File should have a column containing keywords
   - Each row should represent a unique keyword or phrase

4. Select the column containing your keywords

5. Click "Analyze Keywords" to generate:
   - Proximity matrix
   - Visualization heatmap
   - Keyword clusters
   - AI-powered content recommendations

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

See `requirements.txt` for specific versions.

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

## Future Enhancements

- Multiple language support
- Temporal analysis
- Trend detection
- Additional visualization options
- Enhanced export capabilities

## Need Help?

If you encounter any issues or have questions:
1. Check the existing issues on GitHub
2. Create a new issue with a detailed description of your problem
3. Include sample data if possible

## Disclaimer

This tool requires an OpenAI API key and may incur charges based on API usage. Please review OpenAI's pricing before use.
