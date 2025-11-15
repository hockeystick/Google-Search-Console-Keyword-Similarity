# Keyword Similarity Analyzer - macOS Native Application

A native macOS application for analyzing keyword relationships using TF-IDF vectorization and cosine similarity.

## Features

- **Native macOS Interface**: Built with PyQt6 for a native macOS look and feel
- **Tabbed Interface**: Multiple analysis modes in one application
- **Background Processing**: Non-blocking UI with threaded computations
- **Data Visualization**: Interactive heatmaps using matplotlib
- **Export Capabilities**: Export results to CSV
- **Persistent Settings**: Remembers window size and position

## System Requirements

- macOS 10.14 (Mojave) or later
- Python 3.9 or later
- 4GB RAM minimum
- 500MB free disk space

## Installation

### Option 1: Run from Source (Development)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Google-Search-Console-Keyword-Similarity.git
   cd Google-Search-Console-Keyword-Similarity
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements-macos.txt
   ```

   Or use the Makefile:
   ```bash
   make install
   ```

3. **Run the application:**
   ```bash
   python keyword_analyzer_mac.py
   ```

   Or use the Makefile:
   ```bash
   make run
   ```

### Option 2: Build macOS .app Bundle

1. **Install build dependencies:**
   ```bash
   pip install -r requirements-macos.txt
   ```

2. **Build the app:**
   ```bash
   python setup.py py2app
   ```

   Or use the Makefile:
   ```bash
   make build
   ```

3. **Find the app:**
   The built application will be in `dist/Keyword Similarity Analyzer.app`

4. **Install the app:**
   Drag `Keyword Similarity Analyzer.app` to your Applications folder

## Usage

### Basic Similarity Analysis

1. Launch the application
2. Go to the "Basic Analysis" tab
3. Click "Choose CSV File" and select your keyword data
4. Select the column containing keywords
5. Click "Calculate Similarity"
6. Export results using the export buttons

### Visualizing Similarity Matrix

1. Go to the "Visualizer" tab
2. Click "Choose CSV File" and load a pre-computed similarity matrix
3. Use the slider to adjust the number of keywords displayed
4. The heatmap updates in real-time

## Input File Format

Your CSV file should have a column containing keywords:

```csv
keywords
machine learning
deep learning
artificial intelligence
data science
```

For the visualizer, load a CSV file with a similarity matrix:

```csv
,keyword1,keyword2,keyword3
keyword1,1.0,0.85,0.42
keyword2,0.85,1.0,0.38
keyword3,0.42,0.38,1.0
```

## Keyboard Shortcuts

- `⌘ + O` - Open file
- `⌘ + Q` - Quit application
- `⌘ + ,` - Preferences

## Menu Bar

### File
- **Open CSV...** - Open a CSV file
- **Quit** - Exit the application

### Edit
- **Preferences...** - Open preferences (coming soon)

### Help
- **About** - About the application

## Build Commands (Makefile)

The project includes a Makefile for common tasks:

```bash
make install    # Install dependencies
make run        # Run in development mode
make build      # Build .app bundle
make clean      # Clean build artifacts
make test       # Run tests
```

## Troubleshooting

### "App is damaged and can't be opened"

If you see this error when opening the built app:

```bash
xattr -cr "/Applications/Keyword Similarity Analyzer.app"
```

### PyQt6 Import Errors

Make sure you've installed the macOS requirements:

```bash
pip install -r requirements-macos.txt
```

### App Won't Start

Check the Console app for error messages:
1. Open Console.app
2. Search for "Keyword Similarity Analyzer"
3. Look for error messages

## Development

### Project Structure

```
.
├── keyword_analyzer_mac.py    # Main macOS application
├── utils.py                   # Shared utilities
├── config.py                  # Configuration
├── setup.py                   # py2app build script
├── Makefile                   # Build automation
├── requirements-macos.txt     # macOS dependencies
├── tests/                     # Unit tests
└── assets/                    # App icons and resources
```

### Running Tests

```bash
make test
```

Or directly with pytest:

```bash
pytest tests/ -v
```

### Code Style

The project follows PEP 8 guidelines. Format code with:

```bash
black keyword_analyzer_mac.py utils.py config.py
```

## Building for Distribution

To create a distributable .app bundle:

1. **Clean previous builds:**
   ```bash
   make clean
   ```

2. **Build the app:**
   ```bash
   make build
   ```

3. **Test the app:**
   ```bash
   open "dist/Keyword Similarity Analyzer.app"
   ```

4. **Create DMG (optional):**
   ```bash
   hdiutil create -volname "Keyword Analyzer" -srcfolder "dist/Keyword Similarity Analyzer.app" -ov -format UDZO KeywordAnalyzer.dmg
   ```

## Differences from Web Version

The macOS app has several advantages over the Streamlit web version:

- ✅ Native macOS UI (menus, dialogs, etc.)
- ✅ Faster startup (no web server)
- ✅ Offline operation
- ✅ Better file handling
- ✅ Persistent settings
- ✅ Background processing without blocking UI
- ✅ Standard macOS keyboard shortcuts

## Known Limitations

- AI recommendations feature not yet implemented in macOS version
- No real-time collaboration (web version feature)
- Requires local installation

## Roadmap

- [ ] AI-powered recommendations tab
- [ ] Preferences dialog
- [ ] Dark mode support
- [ ] Export to multiple formats (Excel, JSON)
- [ ] Batch processing
- [ ] Keyword clustering visualization
- [ ] Integration with Google Search Console API

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and feature requests, please open an issue on GitHub.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
