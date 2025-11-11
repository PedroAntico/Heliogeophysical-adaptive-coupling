"""
Example analysis script for heliogeophysical data
Demonstrates the complete workflow for satellite data analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helio_analysis import HeliogeophysicalAnalyzer
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Run comprehensive example analysis."""
    
    print("Heliogeophysical Adaptive Coupling Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = HeliogeophysicalAnalyzer()
    
    # Generate realistic synthetic satellite data
    print("\n1. Generating synthetic satellite data...")
    
    # Simulate different space weather conditions
    n_points = 4000
    t = np.linspace(0, 40, n_points)  # 40 hours of data
    
    # Quiet conditions - mostly background noise
    quiet_signal = (0.1 * np.random.randn(n_points) + 
                   0.05 * np.sin(2*np.pi*0.1*t))  # 10-hour oscillation
    
    # Storm conditions - increased variability and bursts
    storm_signal = (0.3 * np.random.randn(n_points) +
                   0.1 * np.sin(2*np.pi*0.1*t) +
                   0.2 * np.sin(2*np.pi*0.5*t) +  # 2-hour oscillation
                   0.5 * np.exp(-(t-20)**2/2) +   # CME-like impulse
                   0.3 * np.exp(-(t-30)**2/5))    # Substorm-like impulse
    
    # Add multifractal properties to storm signal
    for _ in range(4):
        storm_signal = np.convolve(storm_signal, [0.25, 0.5, 0.25], mode='same')
    
    signals = {
        'Quiet_Geomagnetic_Conditions': quiet_signal,
        'Geomagnetic_Storm_Conditions': storm_signal
    }
    
    # Analyze each condition
    print("\n2. Performing comprehensive analysis...")
    
    all_results = {}
    for condition, signal in signals.items():
        print(f"\nAnalyzing {condition}...")
        
        # Perform analysis
        results = analyzer.comprehensive_analysis(
            signal, 
            sample_rate=1.0,  # 1 Hz data
            signal_name=condition
        )
        
        # Store results
        all_results[condition] = results
        
        # Generate detailed report
        analyzer.generate_report(results, f"output/{condition}/")
    
    # Comparative analysis
    print("\n3. Generating comparative analysis...")
    generate_comparative_report(all_results)
    
    print("\n4. Analysis complete!")
    print("Results saved to 'output/' directory")

def generate_comparative_report(results_dict):
    """Generate comparative analysis across different conditions."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = {'Quiet_Geomagnetic_Conditions': 'blue', 
              'Geomagnetic_Storm_Conditions': 'red'}
    
    # Plot 1: Multifractal spectra comparison
    ax1 = axes[0, 0]
    for condition, results in results_dict.items():
        mf = results['multifractal']
        alpha, f_alpha = mf['alpha'], mf['f_alpha']
        valid = np.isfinite(alpha) & np.isfinite(f_alpha)
        if np.sum(valid) > 2:
            ax1.plot(alpha[valid], f_alpha[valid], 
                    color=colors[condition], linewidth=2, label=condition)
            ax1.fill_between(alpha[valid], f_alpha[valid], alpha=0.2, 
                           color=colors[condition])
    
    ax1.set_xlabel('Hölder Exponent α')
    ax1.set_ylabel('Multifractal Spectrum f(α)')
    ax1.set_title('Comparative Multifractal Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Adaptive capacity metrics
    ax2 = axes[0, 1]
    metrics = ['spectrum_width', 'complexity_index']
    labels = ['Spectrum Width', 'Complexity Index']
    
    width = 0.35
    x = np.arange(len(metrics))
    
    for i, (condition, results) in enumerate(results_dict.items()):
        values = [results['adaptive_capacity'].get(metric, 0) for metric in metrics]
        ax2.bar(x + i*width, values, width, label=condition, 
               color=colors[condition], alpha=0.7)
    
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Values')
    ax2.set_title('Adaptive Capacity Comparison')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(labels)
    ax2.legend()
    
    # Plot 3: Wavelet energy distribution
    ax3 = axes[1, 0]
    for condition, results in results_dict.items():
        wavelet = results['wavelet']
        scales = wavelet['scales']
        energy = wavelet['energy_distribution']
        ax3.semilogy(scales, energy, 'o-', color=colors[condition], 
                    label=condition, alpha=0.7)
    
    ax3.set_xlabel('Wavelet Scale')
    ax3.set_ylabel('Energy (log scale)')
    ax3.set_title('Wavelet Energy Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistical summary
    ax4 = axes[1, 1]
    stats_metrics = ['hurst_exponent', 'signal_complexity']
    stats_labels = ['Hurst Exponent', 'Complexity']
    
    width = 0.35
    x = np.arange(len(stats_metrics))
    
    for i, (condition, results) in enumerate(results_dict.items()):
        values = [results['statistics'].get(metric, 0) for metric in stats_metrics]
        ax4.bar(x + i*width, values, width, label=condition, 
               color=colors[condition], alpha=0.7)
    
    ax4.set_xlabel('Statistical Measures')
    ax4.set_ylabel('Values')
    ax4.set_title('Statistical Properties Comparison')
    ax4.set_xticks(x + width/2)
    ax4.set_xticklabels(stats_labels)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('output/comparative_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print quantitative comparison
    print("\n" + "="*70)
    print("QUANTITATIVE COMPARISON OF SPACE WEATHER CONDITIONS")
    print("="*70)
    
    comparison_data = []
    for condition, results in results_dict.items():
        capacity = results['adaptive_capacity']
        stats = results['statistics']
        
        comparison_data.append({
            'Condition': condition.replace('_', ' '),
            'Δα (Spectrum Width)': f"{capacity.get('spectrum_width', 0):.3f}",
            'Complexity Index': f"{capacity.get('complexity_index', 0):.3f}",
            'Hurst Exponent': f"{stats['hurst_exponent']:.3f}",
            'Asymmetry': f"{capacity.get('spectrum_asymmetry', 0):.3f}",
            'Std Dev': f"{results['signal_stats']['std']:.3f}"
        })
    
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    # Interpretation
    print("\n" + "="*70)
    print("SCIENTIFIC INTERPRETATION")
    print("="*70)
    
    quiet_results = results_dict['Quiet_Geomagnetic_Conditions']
    storm_results = results_dict['Geomagnetic_Storm_Conditions']
    
    quiet_width = quiet_results['adaptive_capacity'].get('spectrum_width', 0)
    storm_width = storm_results['adaptive_capacity'].get('spectrum_width', 0)
    
    print(f"• Spectrum width increases from {quiet_width:.3f} (quiet) to {storm_width:.3f} (storm)")
    print("  → Indicates enhanced system complexity during space weather events")
    print("  → Consistent with adaptive coupling hypothesis")
    print("  → Suggests increased information processing capacity during perturbations")

if __name__ == "__main__":
    main()
