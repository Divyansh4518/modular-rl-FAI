"""
Batch testing script - Runs all agents on all mazes automatically.

This script runs comprehensive tests across all combinations of agents
and mazes, generating complete analysis reports.
"""

import os
import time
from datetime import datetime
from test_maze import main as test_maze_main
from test_single_agent import test_agent
import matplotlib.pyplot as plt


def run_batch_tests(output_dir="test_results"):
    """
    Run comprehensive batch tests on all agent-maze combinations.
    
    Args:
        output_dir: Directory to save all test results
    """
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(full_output_dir, exist_ok=True)
    
    print("="*70)
    print("BATCH TESTING - ALL AGENTS ON ALL MAZES")
    print("="*70)
    print(f"\nOutput directory: {full_output_dir}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Test configurations
    maze_configs = [
        ('1', 'Simple', 300, 100),
        ('2', 'Default', 500, 200),
        ('3', 'Complex', 800, 300)
    ]
    
    agents = ['qlearning', 'sarsa', 'expected_sarsa']
    
    total_tests = len(maze_configs) * (1 + len(agents))  # Comparison + individual tests
    test_count = 0
    start_time = time.time()
    
    # Run comparison tests for each maze
    print("\n" + "="*70)
    print("PHASE 1: AGENT COMPARISONS")
    print("="*70 + "\n")
    
    for maze_choice, maze_name, episodes, max_steps in maze_configs:
        test_count += 1
        print(f"\n[{test_count}/{total_tests}] Testing {maze_name} Maze - All Agents Comparison")
        print("-"*70)
        
        test_start = time.time()
        
        # Run comparison test
        test_maze_main(maze_choice=maze_choice, episodes=episodes, max_steps=max_steps)
        
        # Move generated file to output directory
        src_file = "maze_agents_comparison.png"
        if os.path.exists(src_file):
            dst_file = os.path.join(full_output_dir, f"comparison_{maze_name.lower()}.png")
            os.rename(src_file, dst_file)
            print(f"✓ Saved: {dst_file}")
        
        plt.close('all')  # Close all figures
        
        test_time = time.time() - test_start
        print(f"Completed in {test_time:.1f} seconds")
    
    # Run individual agent tests
    print("\n" + "="*70)
    print("PHASE 2: INDIVIDUAL AGENT ANALYSIS")
    print("="*70 + "\n")
    
    for maze_choice, maze_name, episodes, max_steps in maze_configs:
        for agent in agents:
            test_count += 1
            print(f"\n[{test_count}/{total_tests}] Testing {agent.replace('_', ' ').title()} on {maze_name} Maze")
            print("-"*70)
            
            test_start = time.time()
            
            # Run individual agent test
            test_agent(agent_type=agent, maze_choice=maze_choice, episodes=episodes)
            
            # Move generated files
            for file_type in ['learning_curves', 'policy', 'value_function']:
                src_file = f"{agent}_{file_type}.png"
                if os.path.exists(src_file):
                    dst_file = os.path.join(full_output_dir, 
                                           f"{agent}_{maze_name.lower()}_{file_type}.png")
                    os.rename(src_file, dst_file)
                    print(f"✓ Saved: {dst_file}")
            
            plt.close('all')  # Close all figures
            
            test_time = time.time() - test_start
            print(f"Completed in {test_time:.1f} seconds")
    
    # Generate summary report
    total_time = time.time() - start_time
    generate_summary_report(full_output_dir, total_tests, total_time)
    
    print("\n" + "="*70)
    print("BATCH TESTING COMPLETE!")
    print("="*70)
    print(f"\nTotal tests run: {total_tests}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved in: {full_output_dir}")
    print("\nGenerated files:")
    print(f"  - {len(maze_configs)} comparison plots")
    print(f"  - {len(maze_configs) * len(agents) * 3} individual analysis plots")
    print(f"  - 1 summary report (summary_report.txt)")


def generate_summary_report(output_dir, num_tests, total_time):
    """Generate a text summary report of the batch tests."""
    report_path = os.path.join(output_dir, "summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BATCH TESTING SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total tests run: {num_tests}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")
        f.write(f"Average time per test: {total_time/num_tests:.1f} seconds\n\n")
        
        f.write("="*70 + "\n")
        f.write("TESTS PERFORMED\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. Agent Comparisons (3 tests):\n")
        f.write("   - Simple Maze (5x5): Q-Learning vs SARSA vs Expected SARSA\n")
        f.write("   - Default Maze (7x7): Q-Learning vs SARSA vs Expected SARSA\n")
        f.write("   - Complex Maze (10x10): Q-Learning vs SARSA vs Expected SARSA\n\n")
        
        f.write("2. Individual Agent Analysis (9 tests):\n")
        f.write("   Each agent tested on each maze with detailed visualizations:\n")
        f.write("   - Q-Learning: Simple, Default, Complex\n")
        f.write("   - SARSA: Simple, Default, Complex\n")
        f.write("   - Expected SARSA: Simple, Default, Complex\n\n")
        
        f.write("="*70 + "\n")
        f.write("GENERATED FILES\n")
        f.write("="*70 + "\n\n")
        
        f.write("Comparison Plots:\n")
        f.write("  - comparison_simple.png\n")
        f.write("  - comparison_default.png\n")
        f.write("  - comparison_complex.png\n\n")
        
        f.write("Individual Agent Analysis (per agent per maze):\n")
        f.write("  - {agent}_{maze}_learning_curves.png\n")
        f.write("  - {agent}_{maze}_policy.png\n")
        f.write("  - {agent}_{maze}_value_function.png\n\n")
        
        f.write("="*70 + "\n")
        f.write("ANALYSIS RECOMMENDATIONS\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. Compare 'comparison_*.png' files to see relative agent performance\n")
        f.write("2. Check learning curves to identify convergence patterns\n")
        f.write("3. Examine policy visualizations to understand learned strategies\n")
        f.write("4. Review value functions to see state value distributions\n")
        f.write("5. Look for consistency across different maze complexities\n\n")
        
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"\n✓ Summary report saved: {report_path}")


def run_quick_batch(output_dir="quick_test_results"):
    """
    Run a quicker batch test with fewer episodes for rapid validation.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(full_output_dir, exist_ok=True)
    
    print("="*70)
    print("QUICK BATCH TEST - RAPID VALIDATION")
    print("="*70)
    print(f"\nOutput directory: {full_output_dir}\n")
    
    # Quick test configuration
    maze_configs = [
        ('1', 'Simple', 150, 100),
        ('2', 'Default', 250, 200)
    ]
    
    start_time = time.time()
    
    for maze_choice, maze_name, episodes, max_steps in maze_configs:
        print(f"\nQuick test: {maze_name} Maze")
        print("-"*70)
        
        test_maze_main(maze_choice=maze_choice, episodes=episodes, max_steps=max_steps)
        
        src_file = "maze_agents_comparison.png"
        if os.path.exists(src_file):
            dst_file = os.path.join(full_output_dir, f"quick_{maze_name.lower()}.png")
            os.rename(src_file, dst_file)
            print(f"✓ Saved: {dst_file}")
        
        plt.close('all')
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("QUICK BATCH TEST COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Results saved in: {full_output_dir}")


if __name__ == "__main__":
    import sys
    
    print("\nBatch Testing Options:")
    print("  1. Full Batch Test (comprehensive, ~30-45 minutes)")
    print("  2. Quick Batch Test (validation, ~5-10 minutes)")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nSelect option (1/2) [default=2]: ").strip() or '2'
    
    print()
    
    if choice == '1':
        print("Starting FULL batch test...")
        print("This will take approximately 30-45 minutes.\n")
        run_batch_tests()
    else:
        print("Starting QUICK batch test...")
        print("This will take approximately 5-10 minutes.\n")
        run_quick_batch()
