"""
Setup script for creating macOS .app bundle
Usage: python setup.py py2app
"""
import os
from setuptools import setup

APP = ['keyword_analyzer_mac.py']
DATA_FILES = [
    ('.', ['.env.example']),
]

# Check if icon file exists
iconfile_path = 'assets/icon.icns'
iconfile_exists = os.path.exists(iconfile_path)

OPTIONS = {
    'argv_emulation': False,
    'packages': [
        'pandas',
        'numpy',
        'scikit-learn',  # Fixed: was 'sklearn'
        'matplotlib',
        'seaborn',
        'PyQt6',
        'openai',
        'dotenv',
    ],
    'includes': [
        'utils',
        'config',
    ],
    'excludes': [
        'streamlit',
        'tkinter',
    ],
    'plist': {
        'CFBundleName': 'Keyword Similarity Analyzer',
        'CFBundleDisplayName': 'Keyword Similarity Analyzer',
        'CFBundleIdentifier': 'com.keywordanalyzer.app',
        'CFBundleVersion': '2.0.0',
        'CFBundleShortVersionString': '2.0.0',
        'NSHumanReadableCopyright': '© 2024',
        'NSHighResolutionCapable': True,
    },
}

# Only add icon if it exists
if iconfile_exists:
    OPTIONS['iconfile'] = iconfile_path

setup(
    name='Keyword Similarity Analyzer',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
