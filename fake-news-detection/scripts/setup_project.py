# Project setup script
#!/usr/bin/env python3
"""
Setup script for the Fake News Detection project
"""

import os
import sys
from pathlib import Path
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_directory_structure():
    """Create the project directory structure"""
    directories = [
        'data',
        'models',
        'logs',
        'results',
        'src/models',
        'app/templates',
        'scripts',
        'config',
        'notebooks'
    ]
    
    base_dir = Path(__file__).parent.parent
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")

def install_requirements():
    """Install required packages"""
    requirements_file = Path(__file__).parent.parent / 'requirements.txt'
    
    if requirements_file.exists():
        logging.info("Installing requirements...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)])
            logging.info("Requirements installed successfully!")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error installing requirements: {e}")
            return False
    else:
        logging.warning("requirements.txt not found!")
        return False
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    import nltk
    
    nltk_downloads = [
        'punkt',
        'stopwords',
        'wordnet',
        'vader_lexicon'
    ]
    
    for item in nltk_downloads:
        try:
            nltk.download(item, quiet=True)
            logging.info(f"Downloaded NLTK data: {item}")
        except Exception as e:
            logging.warning(f"Could not download {item}: {e}")

def create_sample_data():
    """Create sample data files if they don't exist"""
    import pandas as pd
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    
    # Sample True news
    if not (data_dir / 'True.csv').exists():
        true_news_sample = pd.DataFrame({
            'title': [
                'New Study Shows Benefits of Regular Exercise',
                'Government Announces New Infrastructure Plan',
                'Scientists Discover New Species in Ocean',
                'Local School Receives Education Grant',
                'Weather Service Issues Storm Warning'
            ],
            'text': [
                'According to a comprehensive study published in the Journal of Health Sciences, regular exercise has been shown to improve cardiovascular health and mental well-being. The research, conducted over five years with 10,000 participants, found significant improvements in overall health metrics.',
                'The government today announced a new $50 billion infrastructure plan aimed at improving roads, bridges, and public transportation systems across the country. The plan is expected to create thousands of jobs and improve economic growth.',
                'Marine biologists from the University of California have discovered a new species of deep-sea fish in the Pacific Ocean. The discovery was made during a research expedition using advanced underwater vehicles.',
                'Lincoln Elementary School has received a $100,000 grant from the Department of Education to improve its science and technology programs. The grant will be used to purchase new equipment and train teachers.',
                'The National Weather Service has issued a severe storm warning for the northeastern region, with heavy rain and strong winds expected through the weekend. Residents are advised to take necessary precautions.'
            ],
            'subject': ['health', 'politics', 'science', 'education', 'weather'],
            'date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12']
        })
        
        true_news_sample.to_csv(data_dir / 'True.csv', index=False)
        logging.info("Created sample True.csv file")
    
    # Sample False news
    if not (data_dir / 'False.csv').exists():
        false_news_sample = pd.DataFrame({
            'title': [
                'SHOCKING: Doctors Hate This One Simple Trick!',
                'BREAKING: Government Hiding Alien Contact!',
                'You Won\'t Believe What Happens Next!',
                'URGENT: Share This to Save Lives!',
                'EXPOSED: The Truth They Don\'t Want You to Know!'
            ],
            'text': [
                'This amazing discovery will change your life forever! Doctors are furious about this one simple trick that can cure everything. Big pharma doesn\'t want you to know about this secret method that has been hidden for years!',
                'BREAKING NEWS: Government officials have been secretly meeting with aliens for decades! This shocking revelation will change everything you thought you knew about our world. Share this before it gets deleted!',
                'What happened next will absolutely blow your mind! This incredible story has been shared millions of times and for good reason. You need to see this unbelievable footage right now!',
                'URGENT WARNING: This dangerous substance is in your home right now and could be killing your family! Share this immediately to warn others about this deadly threat that the authorities won\'t tell you about!',
                'The shocking truth about what really happened has finally been exposed! This cover-up goes all the way to the top and they will do anything to keep this information hidden from the public!'
            ],
            'subject': ['health', 'politics', 'entertainment', 'health', 'politics'],
            'date': ['2023-01-10', '2023-02-15', '2023-03-05', '2023-04-01', '2023-05-08']
        })
        
        false_news_sample.to_csv(data_dir / 'False.csv', index=False)
        logging.info("Created sample False.csv file")

def main():
    """Main setup function"""
    logging.info("Starting project setup...")
    
    # Create directory structure
    create_directory_structure()
    
    # Install requirements
    if not install_requirements():
        logging.error("Failed to install requirements. Please install manually.")
        return
    
    # Download NLTK data
    try:
        download_nltk_data()
    except ImportError:
        logging.warning("NLTK not installed. Skipping NLTK data download.")
    
    # Create sample data
    create_sample_data()
    
    logging.info("Project setup completed successfully!")
    logging.info("\nNext steps:")
    logging.info("1. Place your True.csv and False.csv files in the data/ directory")
    logging.info("2. Run: python src/data_preprocessing.py")
    logging.info("3. Run: python src/model_trainer.py")
    logging.info("4. Run: streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()
