"""
Quick test script for maze environment with preset configurations.

This script provides quick testing options with optimized parameters
for different scenarios.
"""

from test_maze import main

def quick_test():
    """Run a quick test with 200 episodes on simple maze."""
    print("QUICK TEST MODE - Simple Maze, 200 episodes")
    print("-" * 70)
    main(maze_choice='1', episodes=200, max_steps=100)


def standard_test():
    """Run standard test with 500 episodes on default maze."""
    print("STANDARD TEST MODE - Default Maze, 500 episodes")
    print("-" * 70)
    main(maze_choice='2', episodes=500, max_steps=200)


def comprehensive_test():
    """Run comprehensive test with 1000 episodes on complex maze."""
    print("COMPREHENSIVE TEST MODE - Complex Maze, 1000 episodes")
    print("-" * 70)
    main(maze_choice='3', episodes=1000, max_steps=300)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("MAZE ENVIRONMENT - QUICK TEST MODES")
    print("="*70)
    print("\nAvailable test modes:")
    print("  1. Quick Test      - Simple Maze (5x5), 200 episodes")
    print("  2. Standard Test   - Default Maze (7x7), 500 episodes")
    print("  3. Comprehensive   - Complex Maze (10x10), 1000 episodes")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nSelect mode (1/2/3) [default=2]: ").strip() or '2'
    
    print()
    
    if choice == '1':
        quick_test()
    elif choice == '3':
        comprehensive_test()
    else:
        standard_test()
