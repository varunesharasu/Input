# Streamlit runner
#!/usr/bin/env python3
"""
Script to run the Streamlit web application
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run Streamlit app"""
    app_path = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\nStreamlit app stopped.")
    except Exception as e:
        print(f"Error running Streamlit app: {e}")

if __name__ == "__main__":
    main()
