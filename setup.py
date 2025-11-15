"""
Setup script for creating macOS .app bundle
Usage: python setup.py py2app
"""
from setuptools import setup

APP = ['keyword_analyzer_mac.py']
DATA_FILES = [
    ('.', ['.env.example']),
]
OPTIONS = {
    'argv_emulation': False,
    'packages': [
        'pandas',
        'numpy',
        'sklearn',
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
    'iconfile': 'assets/icon.icns',  # Add your icon file here
}

setup(
    name='Keyword Similarity Analyzer',
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
