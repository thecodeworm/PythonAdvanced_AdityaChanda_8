"""
Smart Winter Planner - Application Launcher
Run this to start the PyQt GUI application.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.ui.main_window import main
if __name__ == "__main__":
    print("="*70)
    print("SMART WINTER PLANNER")
    print("="*70)
    print("\nStarting GUI application...")
    print("Close the window or press Ctrl+C to exit.\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication closed by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
