import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Calculate pure interaction forces by subtracting baseline data.")
    parser.add_argument("--theta", type=float, required=True, help="Theta angle used in the test.")
    parser.add_argument("--phi", type=float, required=True, help="Phi angle used in the test.")
    parser.add_argument("--plot", action="store_true", help="Plot the resulting interaction forces.")
    
    args = parser.parse_args()

    # Format filenames to match controller_test.py output
    base_name = f"pull_theta{args.theta:.2f}_phi{args.phi:.2f}"
    collect_file = f"{base_name}_raw.csv"
    baseline_file = f"{base_name}_baseline.csv"
    out_file = f"{base_name}_interaction.csv"

    if not os.path.exists(collect_file):
        print(f"Error: Collect file not found -> {collect_file}")
        sys.exit(1)
    if not os.path.exists(baseline_file):
        print(f"Error: Baseline file not found -> {baseline_file}")
        sys.exit(1)

    print(f"Loading {collect_file}...")
    df_collect = pd.read_csv(collect_file, comment="#")
    
    print(f"Loading {baseline_file}...")
    df_baseline = pd.read_csv(baseline_file, comment="#")

    # Ensure lengths match in case one run stopped a few frames early
    min_len = min(len(df_collect), len(df_baseline))
    if len(df_collect) != len(df_baseline):
        print(f"Warning: File lengths differ ({len(df_collect)} vs {len(df_baseline)}). Truncating to {min_len} rows.")
    
    df_collect = df_collect.iloc[:min_len].reset_index(drop=True)
    df_baseline = df_baseline.iloc[:min_len].reset_index(drop=True)

    # Subtract baseline from collect to get pure interaction force
    df_interaction = df_collect - df_baseline

    print(f"Saving interaction forces to {out_file}...")
    df_interaction.to_csv(out_file, index=False)

    # 1. Extract the last line from the collect file
    with open(collect_file, 'r') as f:
        lines = f.readlines()
        metadata_line = lines[-1] if lines else ""

    # 2. Save the interaction dataframe to CSV
    print(f"Saving interaction forces to {out_file}...")
    df_interaction.to_csv(out_file, index=False)

    # 3. Append the extracted metadata line
    if metadata_line.startswith("#"):
        with open(out_file, 'a') as f:
            f.write(metadata_line)

    if args.plot:
        print("Generating plot...")
        plt.figure(figsize=(10, 5))
        
        # Match colors from the original script
        colors = {'Fx': 'r', 'Fy': 'g', 'Fz': 'b', 'Tx': 'yellow', 'Ty': 'teal', 'Tz': 'purple'}
        
        for axis, color in colors.items():
            if axis in df_interaction.columns:
                plt.plot(df_interaction[axis], color=color, alpha=1.0, label=f"Interaction {axis}")
            
        plt.title(f"Pure Interaction Force/Torque (Theta={args.theta:.2f}, Phi={args.phi:.2f})")
        plt.xlabel("Policy Steps (15Hz)")
        plt.ylabel("Force (N) / Torque (Nm)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()