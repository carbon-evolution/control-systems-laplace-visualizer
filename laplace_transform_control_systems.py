import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import control as ctrl
import scipy.signal as signal
from scipy import integrate
import matplotlib.gridspec as gridspec

class LaplaceTransformVisualizer:
    def __init__(self):
        # Define initial system parameters
        self.num = [10]                # Numerator coefficients
        self.den = [1, 2, 10]          # Denominator coefficients (s^2 + 2s + 10)
        self.system_type = 'Second Order'
        
        # Time parameters
        self.t_max = 10
        self.t_samples = 1000
        self.t = np.linspace(0, self.t_max, self.t_samples)
        
        # Frequency parameters
        self.w_max = 20
        self.w_samples = 1000
        self.w = np.logspace(-1, np.log10(self.w_max), self.w_samples)
        
        # Input signal parameters
        self.input_type = 'Step'
        self.input_amplitude = 1.0
        self.input_frequency = 1.0
        
        # Create the control system
        self.update_system()
        
        # Create the main figure
        self.setup_plot()
        
    def update_system(self):
        """Update the control system based on current parameters"""
        # Create transfer function
        self.sys = ctrl.TransferFunction(self.num, self.den)
        
        # Calculate frequency response
        freq_resp = self.sys.frequency_response(self.w)
        self.mag = np.abs(freq_resp.fresp[0]).flatten()
        self.phase = np.angle(freq_resp.fresp[0]).flatten()
        self.omega = freq_resp.omega
        
        # Generate input signal
        if self.input_type == 'Step':
            self.u = np.ones_like(self.t) * self.input_amplitude
            self.u[:10] = 0  # Ensure the step starts slightly after t=0
        elif self.input_type == 'Impulse':
            self.u = np.zeros_like(self.t)
            self.u[10] = self.input_amplitude * self.t_samples / self.t_max  # Scaled impulse
        elif self.input_type == 'Sine':
            self.u = self.input_amplitude * np.sin(2 * np.pi * self.input_frequency * self.t)
        elif self.input_type == 'Ramp':
            self.u = self.input_amplitude * self.t
            self.u[self.t < 0.01] = 0  # Start ramp at t=0
        
        # Calculate time response
        self.t_out, self.y_out = ctrl.forced_response(self.sys, self.t, self.u)
        
        # Calculate the analytical Laplace transform of the input
        self.s = np.complex128(1j) * self.w
        if self.input_type == 'Step':
            self.input_laplace = self.input_amplitude / self.s
        elif self.input_type == 'Impulse':
            self.input_laplace = np.ones_like(self.s) * self.input_amplitude
        elif self.input_type == 'Sine':
            w0 = 2 * np.pi * self.input_frequency
            self.input_laplace = self.input_amplitude * w0 / (self.s**2 + w0**2)
        elif self.input_type == 'Ramp':
            self.input_laplace = self.input_amplitude / (self.s**2)
        
        # Calculate system poles and zeros
        self.poles = ctrl.poles(self.sys)
        self.zeros = ctrl.zeros(self.sys)
    
    def setup_plot(self):
        """Set up the visualization layout"""
        self.fig = plt.figure(figsize=(16, 12))
        # Increase bottom margin to accommodate controls
        gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1, 1, 0.3], hspace=0.4, wspace=0.3)
        
        # Transfer function display
        self.ax_tf = plt.subplot(gs[0, 0])
        self.ax_tf.axis('off')
        self.tf_title = self.ax_tf.set_title("Transfer Function", fontsize=12, pad=20)
        
        # Pole-zero plot
        self.ax_pz = plt.subplot(gs[0, 1])
        self.ax_pz.set_title("Pole-Zero Map", pad=20)
        self.ax_pz.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax_pz.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        self.ax_pz.set_xlabel("Real")
        self.ax_pz.set_ylabel("Imaginary")
        self.ax_pz.grid(True)
        
        # Input signal
        self.ax_input = plt.subplot(gs[1, 0])
        self.ax_input.set_title("Input Signal", pad=20)
        self.ax_input.set_xlabel("Time (s)")
        self.ax_input.set_ylabel("Amplitude")
        self.ax_input.grid(True)
        
        # Step response
        self.ax_response = plt.subplot(gs[1, 1])
        self.ax_response.set_title("System Response", pad=20)
        self.ax_response.set_xlabel("Time (s)")
        self.ax_response.set_ylabel("Amplitude")
        self.ax_response.grid(True)
        
        # Bode magnitude plot
        self.ax_bode_mag = plt.subplot(gs[0:2, 2])
        self.ax_bode_mag.set_title("Bode Magnitude Plot", pad=20)
        self.ax_bode_mag.set_xlabel("Frequency (rad/s)")
        self.ax_bode_mag.set_ylabel("Magnitude (dB)")
        self.ax_bode_mag.grid(True)
        self.ax_bode_mag.set_xscale('log')
        
        # Bode phase plot
        self.ax_bode_phase = plt.subplot(gs[2, 2])
        self.ax_bode_phase.set_title("Bode Phase Plot", pad=20)
        self.ax_bode_phase.set_xlabel("Frequency (rad/s)")
        self.ax_bode_phase.set_ylabel("Phase (degrees)")
        self.ax_bode_phase.grid(True)
        self.ax_bode_phase.set_xscale('log')
        
        # Laplace domain visualization
        self.ax_laplace = plt.subplot(gs[2, 0:2])
        self.ax_laplace.set_title("Laplace Domain: Input * Transfer Function = Output", pad=20)
        self.ax_laplace.set_xlabel("Frequency (rad/s)")
        self.ax_laplace.set_ylabel("Magnitude")
        self.ax_laplace.grid(True)
        self.ax_laplace.set_xscale('log')
        
        # System type selection - moved to bottom left
        self.ax_system_select = plt.axes([0.1, 0.08, 0.15, 0.1])
        self.system_radio = RadioButtons(
            self.ax_system_select,
            ['First Order', 'Second Order', 'PID Controller', 'Resonant System'],
            active=1
        )
        
        # Input type selection - moved to bottom center-left
        self.ax_input_select = plt.axes([0.3, 0.08, 0.15, 0.1])
        self.input_radio = RadioButtons(
            self.ax_input_select,
            ['Step', 'Impulse', 'Sine', 'Ramp'],
            active=0
        )
        
        # Parameter sliders
        self.create_sliders()
        
        # Add a reset button - moved to bottom right
        self.ax_reset = plt.axes([0.85, 0.15, 0.1, 0.04])
        self.reset_button = Button(self.ax_reset, 'Reset')
        self.reset_button.on_clicked(self.reset)
        
        # Connect callbacks
        self.system_radio.on_clicked(self.set_system_type)
        self.input_radio.on_clicked(self.set_input_type)
        
        # Initial update
        self.update_plots()
        
        # Adjust subplot parameters to give specified padding
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.25)
    
    def create_sliders(self):
        """Create parameter sliders based on system type"""
        self.sliders = {}
        
        # Remove existing sliders if any
        if hasattr(self, 'slider_axes'):
            for ax in self.slider_axes:
                self.fig.delaxes(ax)
        
        self.slider_axes = []
        
        # Base positions for sliders
        slider_left = 0.5  # Start sliders more to the right
        slider_width = 0.3  # Make sliders wider
        slider_height = 0.02
        base_bottom = 0.15  # Move sliders up slightly
        vertical_spacing = 0.03  # Space between sliders
        
        if self.system_type == 'First Order':
            # K/(Ts + 1)
            ax_gain = plt.axes([slider_left, base_bottom + vertical_spacing, slider_width, slider_height])
            self.sliders['gain'] = Slider(ax_gain, 'Gain (K)', 0.1, 10.0, valinit=1, valstep=0.1)
            self.slider_axes.append(ax_gain)
            
            ax_time_const = plt.axes([slider_left, base_bottom, slider_width, slider_height])
            self.sliders['time_constant'] = Slider(ax_time_const, 'Time Constant (T)', 0.1, 5.0, valinit=1, valstep=0.1)
            self.slider_axes.append(ax_time_const)
            
        elif self.system_type == 'Second Order':
            # K/(s^2 + 2*zeta*wn*s + wn^2)
            ax_wn = plt.axes([slider_left, base_bottom + 2*vertical_spacing, slider_width, slider_height])
            self.sliders['natural_freq'] = Slider(ax_wn, 'Natural Freq (ωn)', 0.1, 10.0, valinit=3.16, valstep=0.1)
            self.slider_axes.append(ax_wn)
            
            ax_gain = plt.axes([slider_left, base_bottom + vertical_spacing, slider_width, slider_height])
            self.sliders['gain'] = Slider(ax_gain, 'Gain (K)', 0.1, 20.0, valinit=10, valstep=0.1)
            self.slider_axes.append(ax_gain)
            
            ax_zeta = plt.axes([slider_left, base_bottom, slider_width, slider_height])
            self.sliders['damping'] = Slider(ax_zeta, 'Damping Ratio (ζ)', 0.01, 2.0, valinit=0.2, valstep=0.01)
            self.slider_axes.append(ax_zeta)
            
        elif self.system_type == 'PID Controller':
            # Kp + Ki/s + Kd*s
            ax_kp = plt.axes([slider_left, base_bottom + 2*vertical_spacing, slider_width, slider_height])
            self.sliders['kp'] = Slider(ax_kp, 'Proportional (Kp)', 0.0, 10.0, valinit=1.0, valstep=0.1)
            self.slider_axes.append(ax_kp)
            
            ax_ki = plt.axes([slider_left, base_bottom + vertical_spacing, slider_width, slider_height])
            self.sliders['ki'] = Slider(ax_ki, 'Integral (Ki)', 0.0, 5.0, valinit=0.5, valstep=0.1)
            self.slider_axes.append(ax_ki)
            
            ax_kd = plt.axes([slider_left, base_bottom, slider_width, slider_height])
            self.sliders['kd'] = Slider(ax_kd, 'Derivative (Kd)', 0.0, 2.0, valinit=0.2, valstep=0.01)
            self.slider_axes.append(ax_kd)
            
        elif self.system_type == 'Resonant System':
            # K*s/(s^2 + 2*zeta*wn*s + wn^2)
            ax_gain = plt.axes([slider_left, base_bottom + 2*vertical_spacing, slider_width, slider_height])
            self.sliders['gain'] = Slider(ax_gain, 'Gain (K)', 0.1, 10.0, valinit=1.0, valstep=0.1)
            self.slider_axes.append(ax_gain)
            
            ax_zeta = plt.axes([slider_left, base_bottom + vertical_spacing, slider_width, slider_height])
            self.sliders['damping'] = Slider(ax_zeta, 'Damping Ratio (ζ)', 0.01, 1.0, valinit=0.1, valstep=0.01)
            self.slider_axes.append(ax_zeta)
            
            ax_wn = plt.axes([slider_left, base_bottom, slider_width, slider_height])
            self.sliders['natural_freq'] = Slider(ax_wn, 'Resonance Freq (ωn)', 0.1, 10.0, valinit=5.0, valstep=0.1)
            self.slider_axes.append(ax_wn)
        
        # Input parameter sliders - always show amplitude
        ax_amplitude = plt.axes([slider_left, base_bottom - vertical_spacing, slider_width, slider_height])
        self.sliders['input_amplitude'] = Slider(ax_amplitude, 'Input Amplitude', 0.1, 5.0, valinit=1.0, valstep=0.1)
        self.slider_axes.append(ax_amplitude)
        
        # Add frequency slider for sine input
        if self.input_type == 'Sine':
            ax_freq = plt.axes([slider_left, base_bottom - 2*vertical_spacing, slider_width, slider_height])
            self.sliders['input_frequency'] = Slider(ax_freq, 'Input Frequency (Hz)', 0.1, 5.0, valinit=1.0, valstep=0.1)
            self.slider_axes.append(ax_freq)
        
        # Connect callbacks
        for name, slider in self.sliders.items():
            slider.on_changed(self.update_on_slider)
    
    def set_system_type(self, label):
        """Change the type of control system"""
        self.system_type = label
        
        # Update numerator and denominator based on system type
        if label == 'First Order':
            self.num = [1]
            self.den = [1, 1]
        elif label == 'Second Order':
            self.num = [10]
            self.den = [1, 2, 10]
        elif label == 'PID Controller':
            # Modified PID to be proper transfer function
            # (Kd*s^2 + Kp*s + Ki)/(s^2 + 1000s)
            self.num = [0.2, 1, 0.5]
            self.den = [1, 1000, 0]  # s(s + 1000)
        elif label == 'Resonant System':
            # Modified resonant system to be proper
            # K*s/(s^2 + 2*zeta*wn*s + wn^2)
            self.num = [1, 0]  # K*s
            self.den = [1, 0.2, 25]  # s^2 + 2*zeta*wn*s + wn^2
        
        # Recreate sliders and update plots
        self.create_sliders()
        self.update_system()
        self.update_plots()
    
    def set_input_type(self, label):
        """Change the input signal type"""
        self.input_type = label
        
        # Recreate sliders for the new input type
        self.create_sliders()
        self.update_system()
        self.update_plots()
    
    def update_on_slider(self, val):
        """Update system based on slider changes"""
        # Update system parameters based on sliders
        if self.system_type == 'First Order':
            gain = self.sliders['gain'].val
            time_constant = self.sliders['time_constant'].val
            self.num = [gain]
            self.den = [time_constant, 1]
            
        elif self.system_type == 'Second Order':
            gain = self.sliders['gain'].val
            zeta = self.sliders['damping'].val
            wn = self.sliders['natural_freq'].val
            self.num = [gain]
            self.den = [1, 2*zeta*wn, wn**2]
            
        elif self.system_type == 'PID Controller':
            kp = self.sliders['kp'].val
            ki = self.sliders['ki'].val
            kd = self.sliders['kd'].val
            self.num = [kd, kp, ki]  # Kd*s^2 + Kp*s + Ki
            self.den = [1, 0]  # s
            
        elif self.system_type == 'Resonant System':
            gain = self.sliders['gain'].val
            zeta = self.sliders['damping'].val
            wn = self.sliders['natural_freq'].val
            self.num = [gain, 0]  # K*s
            self.den = [1, 2*zeta*wn, wn**2]  # s^2 + 2*zeta*wn*s + wn^2
        
        # Update input parameters
        self.input_amplitude = self.sliders['input_amplitude'].val
        if self.input_type == 'Sine' and 'input_frequency' in self.sliders:
            self.input_frequency = self.sliders['input_frequency'].val
        
        # Update system and plots
        self.update_system()
        self.update_plots()
    
    def reset(self, event):
        """Reset all parameters to default values"""
        for slider in self.sliders.values():
            slider.reset()
        self.update_on_slider(None)
    
    def update_plots(self):
        """Update all plots with current system data"""
        # Clear all axes
        for ax in [self.ax_pz, self.ax_input, self.ax_response, 
                  self.ax_bode_mag, self.ax_bode_phase, self.ax_laplace]:
            ax.clear()
            ax.grid(True)
        
        # Update transfer function display
        self.ax_tf.clear()
        self.ax_tf.axis('off')
        
        # Format transfer function strings
        if self.system_type == 'First Order':
            tf_str = f"$G(s) = \\frac{{{self.num[0]}}}{{{self.den[0]}s + {self.den[1]}}}$"
        elif self.system_type == 'Second Order':
            tf_str = f"$G(s) = \\frac{{{self.num[0]}}}{{s^2 + {self.den[1]}s + {self.den[2]}}}$"
        elif self.system_type == 'PID Controller':
            tf_str = f"$G(s) = \\frac{{{self.num[0]}s^2 + {self.num[1]}s + {self.num[2]}}}{{s^2 + {self.den[1]}s}}$"
        elif self.system_type == 'Resonant System':
            tf_str = f"$G(s) = \\frac{{{self.num[0]}s}}{{s^2 + {self.den[1]}s + {self.den[2]}}}$"
        
        self.ax_tf.text(0.5, 0.6, tf_str, size=14, ha='center', va='center')
        
        # Display poles and zeros
        if len(self.poles) > 0:
            pole_str = "Poles: " + ", ".join([f"{p:.2f}" for p in self.poles])
            self.ax_tf.text(0.5, 0.3, pole_str, size=10, ha='center', va='center')
        
        if len(self.zeros) > 0:
            zero_str = "Zeros: " + ", ".join([f"{z:.2f}" for z in self.zeros])
            self.ax_tf.text(0.5, 0.1, zero_str, size=10, ha='center', va='center')
        
        # Update pole-zero plot
        self.ax_pz.plot(np.real(self.poles), np.imag(self.poles), 'rx', ms=10, label='Poles')
        if len(self.zeros) > 0:
            self.ax_pz.plot(np.real(self.zeros), np.imag(self.zeros), 'bo', ms=10, label='Zeros')
        self.ax_pz.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax_pz.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        self.ax_pz.set_xlabel("Real")
        self.ax_pz.set_ylabel("Imaginary")
        self.ax_pz.set_title("Pole-Zero Map")
        self.ax_pz.legend()
        
        # Update input signal plot
        self.ax_input.plot(self.t, self.u, 'b-')
        self.ax_input.set_xlabel("Time (s)")
        self.ax_input.set_ylabel("Amplitude")
        self.ax_input.set_title(f"{self.input_type} Input")
        
        # Update response plot
        self.ax_response.plot(self.t_out, self.y_out, 'r-')
        self.ax_response.set_xlabel("Time (s)")
        self.ax_response.set_ylabel("Amplitude")
        self.ax_response.set_title("System Response")
        
        # Update Bode magnitude plot
        self.ax_bode_mag.semilogx(self.omega, 20 * np.log10(self.mag), 'g-')
        self.ax_bode_mag.set_xlabel("Frequency (rad/s)")
        self.ax_bode_mag.set_ylabel("Magnitude (dB)")
        self.ax_bode_mag.set_title("Bode Magnitude Plot")
        
        # Update Bode phase plot
        self.ax_bode_phase.semilogx(self.omega, np.rad2deg(self.phase), 'g-')
        self.ax_bode_phase.set_xlabel("Frequency (rad/s)")
        self.ax_bode_phase.set_ylabel("Phase (degrees)")
        self.ax_bode_phase.set_title("Bode Phase Plot")
        
        # Update Laplace domain visualization
        # Calculate system response in frequency domain
        tf_vals = ctrl.evalfr(self.sys, self.s)
        output_laplace = self.input_laplace * tf_vals
        
        # Plot the input, system TF, and output in the Laplace domain
        self.ax_laplace.loglog(self.w, np.abs(self.input_laplace), 'b-', label='Input')
        self.ax_laplace.loglog(self.w, np.abs(tf_vals), 'g-', label='Transfer Function')
        self.ax_laplace.loglog(self.w, np.abs(output_laplace), 'r-', label='Output')
        self.ax_laplace.set_xlabel("Frequency (rad/s)")
        self.ax_laplace.set_ylabel("Magnitude")
        self.ax_laplace.set_title("Laplace Domain: Input * Transfer Function = Output")
        self.ax_laplace.legend()
        
        plt.draw()

def main():
    # Check if control package is available
    try:
        import control
    except ImportError:
        print("This program requires the 'control' package.")
        print("Please install it using: pip install control")
        return
        
    visualizer = LaplaceTransformVisualizer()
    plt.show()

if __name__ == "__main__":
    main()