
##  HELIO_ANALYSIS.PY

```python
"""
Heliogeophysical Adaptive Coupling Analysis
Core implementation of wavelet and multifractal methods for solar-terrestrial research

Author: Pedro Guilherme Antico
Date: November 2025
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import pywt
from sklearn.metrics import mean_squared_error
import pandas as pd
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class HeliogeophysicalAnalyzer:
    """
    Integrated analyzer for heliogeophysical time series using wavelet 
    and multifractal methods to quantify adaptive coupling characteristics.
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the analyzer with default parameters.
        
        Parameters:
        -----------
        random_seed : int
            Random seed for reproducible results
        """
        np.random.seed(random_seed)
        self.results = {}
        
    def continuous_wavelet_transform(self, signal: np.ndarray, scales: np.ndarray, 
                                   wavelet: str = 'cmor1.5-1.0') -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Continuous Wavelet Transform (CWT) for time-frequency analysis.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input time series
        scales : np.ndarray  
            Wavelet scales to analyze
        wavelet : str
            Wavelet type (default: complex morlet)
            
        Returns:
        --------
        coefficients : np.ndarray
            CWT coefficients (complex)
        frequencies : np.ndarray
            Pseudo-frequencies for each scale
        """
        # Perform CWT
        coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
        
        return coefficients, frequencies
    
    def wavelet_coherence(self, x: np.ndarray, y: np.ndarray, scales: np.ndarray,
                         wavelet: str = 'cmor1.5-1.0') -> np.ndarray:
        """
        Calculate wavelet coherence between two time series.
        
        Parameters:
        -----------
        x, y : np.ndarray
            Input time series
        scales : np.ndarray
            Wavelet scales
        wavelet : str
            Wavelet type
            
        Returns:
        --------
        coherence : np.ndarray
            Wavelet coherence matrix
        """
        # Compute CWT for both signals
        Wx, _ = self.continuous_wavelet_transform(x, scales, wavelet)
        Wy, _ = self.continuous_wavelet_transform(y, scales, wavelet)
        
        # Smoothing function
        def smooth(C, scale):
            k = 2 * int(scale) + 1
            return np.array([np.convolve(c, np.ones(k)/k, mode='same') for c in C])
        
        # Calculate coherence
        Wxy = Wx * np.conj(Wy)
        Sxy = smooth(Wxy, 0.6)
        Sxx = smooth(np.abs(Wx)**2, 0.6)
        Syy = smooth(np.abs(Wy)**2, 0.6)
        
        coherence = np.abs(Sxy)**2 / (Sxx * Syy)
        
        return coherence
    
    def mfdfa(self, signal: np.ndarray, scales: np.ndarray, q: np.ndarray = None, 
              order: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Multifractal Detrended Fluctuation Analysis (MF-DFA).
        
        Parameters:
        -----------
        signal : np.ndarray
            Input time series
        scales : np.ndarray
            Window sizes for analysis
        q : np.ndarray
            Moment orders (if None, uses range -5 to 5)
        order : int
            Detrending polynomial order
            
        Returns:
        --------
        scales : np.ndarray
            Analysis scales
        Fq : np.ndarray
            Fluctuation functions for each q
        hq : np.ndarray
            Generalized Hurst exponents
        """
        if q is None:
            q = np.linspace(-5, 5, 21)
        
        N = len(signal)
        # Integrated signal
        y = np.cumsum(signal - np.mean(signal))
        
        Fq = np.zeros((len(scales), len(q)))
        
        for i, scale in enumerate(scales):
            # Segment signal
            segments = int(np.floor(N / scale))
            if segments < 2:
                continue
                
            # Vectorized segmentation
            indices = np.arange(segments * scale, dtype=int)
            reshaped = y[indices].reshape(segments, int(scale))
            
            # Local trends
            x = np.arange(scale)
            trends = np.polyfit(x, reshaped.T, order)
            if order == 1:
                trends = trends[0] * x[:, None] + trends[1]
            else:
                # Higher order polynomial fitting
                trends = np.zeros_like(reshaped.T)
                for j in range(segments):
                    p = np.poly1d(trends[:, j])
                    trends[:, j] = p(x)
            
            # Fluctuations
            fluctuations = reshaped.T - trends
            variances = np.mean(fluctuations**2, axis=0)
            
            # q-th order fluctuation function
            for j, q_val in enumerate(q):
                if q_val == 0:
                    Fq[i, j] = np.exp(0.5 * np.mean(np.log(variances)))
                else:
                    Fq[i, j] = np.mean(variances**(q_val/2))**(1/q_val)
        
        # Calculate Hurst exponents
        hq = np.zeros(len(q))
        for j in range(len(q)):
            valid = Fq[:, j] > 0
            if np.sum(valid) > 2:
                coeffs = np.polyfit(np.log(scales[valid]), np.log(Fq[valid, j]), 1)
                hq[j] = coeffs[0]
        
        return scales, Fq, hq
    
    def multifractal_spectrum(self, hq: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate multifractal spectrum from generalized Hurst exponents.
        
        Parameters:
        -----------
        hq : np.ndarray
            Generalized Hurst exponents
        q : np.ndarray
            Moment orders
            
        Returns:
        --------
        alpha : np.ndarray
            Hölder exponents
        f_alpha : np.ndarray
            Multifractal spectrum
        """
        # Legendre transform
        tau = q * hq - 1
        alpha = np.gradient(tau, q)
        f_alpha = q * alpha - tau
        
        return alpha, f_alpha
    
    def calculate_adaptive_capacity(self, alpha: np.ndarray, f_alpha: np.ndarray) -> Dict[str, float]:
        """
        Calculate adaptive capacity metrics from multifractal spectrum.
        
        Parameters:
        -----------
        alpha : np.ndarray
            Hölder exponents
        f_alpha : np.ndarray
            Multifractal spectrum
            
        Returns:
        --------
        metrics : Dict
            Dictionary of adaptive capacity metrics
        """
        # Remove invalid values
        valid = np.isfinite(alpha) & np.isfinite(f_alpha)
        alpha_clean = alpha[valid]
        f_alpha_clean = f_alpha[valid]
        
        if len(alpha_clean) < 3:
            return {}
        
        # Spectrum width (complexity measure)
        spectrum_width = np.max(alpha_clean) - np.min(alpha_clean)
        
        # Spectrum asymmetry
        alpha_max = alpha_clean[np.argmax(f_alpha_clean)]
        alpha_range = alpha_clean[-1] - alpha_clean[0]
        asymmetry = (alpha_max - alpha_clean[0]) / alpha_range if alpha_range > 0 else 0
        
        # Complexity index (combination of width and structure)
        complexity_index = spectrum_width * (1 + np.abs(asymmetry))
        
        metrics = {
            'spectrum_width': spectrum_width,
            'spectrum_asymmetry': asymmetry,
            'complexity_index': complexity_index,
            'alpha_range': [np.min(alpha_clean), np.max(alpha_clean)],
            'f_alpha_range': [np.min(f_alpha_clean), np.max(f_alpha_clean)]
        }
        
        return metrics
    
    def generate_synthetic_signal(self, n_points: int = 1000, signal_type: str = 'multifractal') -> np.ndarray:
        """
        Generate synthetic signals for testing and demonstration.
        
        Parameters:
        -----------
        n_points : int
            Length of generated signal
        signal_type : str
            Type of signal ('multifractal', 'periodic', 'chaotic')
            
        Returns:
        --------
        signal : np.ndarray
            Generated time series
        """
        t = np.linspace(0, 10, n_points)
        
        if signal_type == 'multifractal':
            # Generate multifractal-like signal using cascading process
            signal = np.random.randn(n_points)
            for _ in range(6):
                signal = np.repeat(signal, 2)
                signal = signal[:n_points]
                signal += 0.5 * np.random.randn(n_points)
            signal = signal[:n_points]
            
        elif signal_type == 'periodic':
            # Mixed periodic signals
            signal = (np.sin(2*np.pi*2*t) + 0.5*np.sin(2*np.pi*5*t) + 
                      0.3*np.sin(2*np.pi*13*t))
            
        elif signal_type == 'chaotic':
            # Logistic map for chaotic behavior
            signal = np.zeros(n_points)
            signal[0] = 0.5
            for i in range(1, n_points):
                signal[i] = 3.9 * signal[i-1] * (1 - signal[i-1])
                
        else:
            signal = np.random.randn(n_points)
        
        return signal - np.mean(signal)
    
    def comprehensive_analysis(self, signal: np.ndarray, sample_rate: float = 1.0, 
                             signal_name: str = "Unknown") -> Dict:
        """
        Perform comprehensive heliogeophysical analysis on input signal.
        
        Parameters:
        -----------
        signal : np.ndarray
            Input time series
        sample_rate : float
            Sampling frequency (Hz)
        signal_name : str
            Identifier for the signal
            
        Returns:
        --------
        results : Dict
            Comprehensive analysis results
        """
        print(f"Performing comprehensive analysis on {signal_name}...")
        
        # Analysis parameters
        scales = np.arange(8, 128, 4)
        q_orders = np.linspace(-5, 5, 21)
        mfdfa_scales = np.logspace(np.log10(16), np.log10(len(signal)//4), 20).astype(int)
        
        results = {
            'signal_name': signal_name,
            'signal_stats': {
                'length': len(signal),
                'mean': np.mean(signal),
                'std': np.std(signal),
                'kurtosis': stats.kurtosis(signal),
                'skewness': stats.skew(signal)
            }
        }
        
        # 1. Wavelet Analysis
        print("  Performing wavelet analysis...")
        wavelet_coeffs, frequencies = self.continuous_wavelet_transform(signal, scales)
        results['wavelet'] = {
            'coefficients': wavelet_coeffs,
            'scales': scales,
            'frequencies': frequencies,
            'energy_distribution': np.mean(np.abs(wavelet_coeffs)**2, axis=1)
        }
        
        # 2. Multifractal Analysis
        print("  Performing multifractal analysis...")
        mf_scales, Fq, hq = self.mfdfa(signal, mfdfa_scales, q_orders)
        alpha, f_alpha = self.multifractal_spectrum(hq, q_orders)
        
        results['multifractal'] = {
            'scales': mf_scales,
            'Fq': Fq,
            'hq': hq,
            'alpha': alpha,
            'f_alpha': f_alpha,
            'q_orders': q_orders
        }
        
        # 3. Adaptive Capacity Metrics
        print("  Calculating adaptive capacity metrics...")
        capacity_metrics = self.calculate_adaptive_capacity(alpha, f_alpha)
        results['adaptive_capacity'] = capacity_metrics
        
        # 4. Statistical Analysis
        print("  Performing statistical analysis...")
        # Calculate Hurst exponent (monofractal approximation)
        hurst = hq[np.abs(q_orders).argmin()]  # h(q=2) approximation
        results['statistics'] = {
            'hurst_exponent': hurst,
            'signal_complexity': capacity_metrics.get('complexity_index', 0),
            'entropy_approx': np.std(signal) / np.mean(np.abs(np.diff(signal))) if np.mean(np.abs(np.diff(signal))) > 0 else 0
        }
        
        print(f"  Analysis complete. Complexity index: {capacity_metrics.get('complexity_index', 0):.3f}")
        
        self.results[signal_name] = results
        return results
    
    def generate_report(self, results: Dict, output_dir: str = "./"):
        """
        Generate comprehensive visualization report.
        
        Parameters:
        -----------
        results : Dict
            Analysis results from comprehensive_analysis
        output_dir : str
            Output directory for figures
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        signal_name = results['signal_name']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Original signal
        ax1 = plt.subplot(3, 3, 1)
        signal_stats = results['signal_stats']
        plt.plot(np.arange(len(results.get('signal', np.zeros(1000)))/1000, 
                 results.get('signal', np.zeros(1000)))
        plt.title(f'Time Series: {signal_name}\n'
                 f'Length: {signal_stats["length"]}, '
                 f'Std: {signal_stats["std"]:.3f}')
        plt.xlabel('Time (arb. units)')
        plt.ylabel('Amplitude')
        
        # 2. Wavelet scalogram
        ax2 = plt.subplot(3, 3, 2)
        wavelet_results = results['wavelet']
        coeffs = wavelet_results['coefficients']
        plt.imshow(np.abs(coeffs), extent=[0, 1, wavelet_results['scales'][-1], wavelet_results['scales'][0]], 
                  aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude')
        plt.title('Wavelet Scalogram')
        plt.ylabel('Scale')
        plt.xlabel('Time')
        
        # 3. Multifractal spectrum
        ax3 = plt.subplot(3, 3, 3)
        mf_results = results['multifractal']
        alpha, f_alpha = mf_results['alpha'], mf_results['f_alpha']
        valid = np.isfinite(alpha) & np.isfinite(f_alpha)
        if np.sum(valid) > 2:
            plt.plot(alpha[valid], f_alpha[valid], 'b-', linewidth=2, label='Multifractal spectrum')
            plt.fill_between(alpha[valid], f_alpha[valid], alpha=0.3)
            
            capacity_metrics = results['adaptive_capacity']
            spectrum_width = capacity_metrics.get('spectrum_width', 0)
            plt.title(f'Multifractal Spectrum\nWidth: {spectrum_width:.3f}')
            plt.xlabel('Hölder exponent α')
            plt.ylabel('f(α)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. Fluctuation functions
        ax4 = plt.subplot(3, 3, 4)
        Fq = mf_results['Fq']
        scales = mf_results['scales']
        q_orders = mf_results['q_orders']
        
        for i, q in enumerate(q_orders[::4]):  # Plot every 4th q for clarity
            if np.all(Fq[:, i] > 0):
                plt.loglog(scales, Fq[:, i], 'o-', label=f'q = {q:.1f}')
        
        plt.title('MF-DFA Fluctuation Functions')
        plt.xlabel('Scale')
        plt.ylabel('F(q)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Hurst exponents
        ax5 = plt.subplot(3, 3, 5)
        hq = mf_results['hq']
        valid_hq = np.isfinite(hq)
        if np.sum(valid_hq) > 2:
            plt.plot(q_orders[valid_hq], hq[valid_hq], 'ro-', linewidth=2)
            plt.title(f'Generalized Hurst Exponents\nH(2) = {results["statistics"]["hurst_exponent"]:.3f}')
            plt.xlabel('Moment q')
            plt.ylabel('h(q)')
            plt.grid(True, alpha=0.3)
        
        # 6. Adaptive capacity metrics
        ax6 = plt.subplot(3, 3, 6)
        capacity_metrics = results['adaptive_capacity']
        if capacity_metrics:
            metrics_to_plot = ['spectrum_width', 'complexity_index', 'spectrum_asymmetry']
            values = [capacity_metrics.get(m, 0) for m in metrics_to_plot]
            labels = ['Spectrum Width', 'Complexity Index', 'Asymmetry']
            
            bars = plt.bar(labels, values, alpha=0.7, color=['#2E86AB', '#A23B72', '#F18F01'])
            plt.title('Adaptive Capacity Metrics')
            plt.ylabel('Metric Value')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 7. Energy distribution across scales
        ax7 = plt.subplot(3, 3, 7)
        energy = wavelet_results['energy_distribution']
        plt.plot(wavelet_results['scales'], energy, 'g-', linewidth=2)
        plt.title('Wavelet Energy Distribution')
        plt.xlabel('Scale')
        plt.ylabel('Energy')
        plt.grid(True, alpha=0.3)
        
        # 8. Statistical summary
        ax8 = plt.subplot(3, 3, 8)
        stats_data = results['statistics']
        stat_names = ['Hurst Exponent', 'Signal Complexity', 'Approx. Entropy']
        stat_values = [stats_data['hurst_exponent'], stats_data['signal_complexity'], 
                      stats_data['entropy_approx']]
        
        bars = plt.bar(stat_names, stat_values, alpha=0.7, color=['#1B998B', '#ED217C', '#2D3047'])
        plt.title('Statistical Summary')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        for bar, value in zip(bars, stat_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 9. Signal histogram and distribution
        ax9 = plt.subplot(3, 3, 9)
        signal_data = results.get('signal', np.random.randn(1000))
        plt.hist(signal_data, bins=50, density=True, alpha=0.7, color='purple')
        plt.title('Signal Distribution')
        plt.xlabel('Amplitude')
        plt.ylabel('Density')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{signal_name}_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Report generated: {output_dir}/{signal_name}_comprehensive_analysis.png")
        
        # Save numerical results
        self.save_numerical_results(results, output_dir)
    
    def save_numerical_results(self, results: Dict, output_dir: str):
        """
        Save numerical results to CSV files.
        
        Parameters:
        -----------
        results : Dict
            Analysis results
        output_dir : str
            Output directory
        """
        signal_name = results['signal_name']
        
        # Save multifractal spectrum
        mf_data = pd.DataFrame({
            'alpha': results['multifractal']['alpha'],
            'f_alpha': results['multifractal']['f_alpha'],
            'q_orders': results['multifractal']['q_orders'],
            'hq': results['multifractal']['hq']
        })
        mf_data.to_csv(f'{output_dir}/{signal_name}_multifractal_spectrum.csv', index=False)
        
        # Save adaptive capacity metrics
        capacity_data = pd.DataFrame([results['adaptive_capacity']])
        capacity_data.to_csv(f'{output_dir}/{signal_name}_adaptive_capacity.csv', index=False)
        
        # Save statistical summary
        stats_data = pd.DataFrame([results['statistics']])
        stats_data.to_csv(f'{output_dir}/{signal_name}_statistics.csv', index=False)
        
        print(f"Numerical results saved to {output_dir}")

# Example usage and testing
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = HeliogeophysicalAnalyzer()
    
    # Generate test signals
    print("Generating test signals...")
    multifractal_signal = analyzer.generate_synthetic_signal(2000, 'multifractal')
    periodic_signal = analyzer.generate_synthetic_signal(2000, 'periodic')
    chaotic_signal = analyzer.generate_synthetic_signal(2000, 'chaotic')
    
    # Analyze each signal
    signals = {
        'Multifractal_Signal': multifractal_signal,
        'Periodic_Signal': periodic_signal, 
        'Chaotic_Signal': chaotic_signal
    }
    
    all_results = {}
    for name, signal in signals.items():
        results = analyzer.comprehensive_analysis(signal, signal_name=name)
        analyzer.generate_report(results, "example_output/")
        all_results[name] = results
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    summary_data = []
    for name, results in all_results.items():
        capacity = results['adaptive_capacity']
        stats = results['statistics']
        
        summary_data.append({
            'Signal': name,
            'Spectrum Width': f"{capacity.get('spectrum_width', 0):.3f}",
            'Complexity Index': f"{capacity.get('complexity_index', 0):.3f}",
            'Hurst Exponent': f"{stats['hurst_exponent']:.3f}",
            'Asymmetry': f"{capacity.get('spectrum_asymmetry', 0):.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
