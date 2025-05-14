# Vibrating String Simulation
A Python package to simulate a vibrating string using Pygame for visualization and SciPy for physics calculations. This tool models a string fixed at one end and driven at the other, with interactive controls to adjust parameters like frequency, damping, and driving force.

## Usage
### Command Line
```bash
python vibrating-string.py
```
### Python Script
```python
if __name__ == "__main__":
    sim = StringSimulation(
        N=100, L=1.0, Mtot=0.8, T=5.0, b=0.02, F0=1.0, f=None,
        width=800, height=600, left_margin=50, right_margin=50, top_margin=100, bottom_margin=100
    )
    sim.run()
```
## Controls
- **UP**: Increase frequency (`f`)
- **DOWN**: Decrease frequency (`f`)
- **LEFT**: Decrease time step (`dt`)
- **RIGHT**: Increase time step (`dt`)
- **Z**: Decrease damping (`b`)
- **X**: Increase damping (`b`)
- **K**: Decrease driving force (`F0`)
- **L**: Increase driving force (`F0`)
- **H**: Toggle help menu
- **S**: Toggle sticks (end supports)
- **J**: Toggle shadow
- **SPACE**: Pause/resume
- **R**: Reset simulation
- **F**: Toggle driving force on/off
- **D**: Toggle spheres (markers)
- **NUMPAD +/-**: Increase/decrease number of spheres
- **P**: Print simulation log to file
- **G**: Toggle axes
- **Y**: Toggle fixed parameters display
- **1-9**: Set frequency to \( n \times f_0 \) (normal modes)
## Parameters
- `N`: Number of segments (default: 100)
- `L`: String length (m, default: 1.0)
- `Mtot`: Mass of the string (kg, default: 0.8)
- `T`: Tension (N, default: 5.0)
- `b`: Damping coefficient (kg/s, default: 0.01)
- `F0`: Driving force amplitude (N, default: 1.0)
- `f`: Initial frequency (Hz, default: fundamental frequency \( f_0 \))
- `width`, `height`: Window size (pixels, default: 800x600)
- `left_margin`, `right_margin`, `top_margin`, `bottom_margin`: Screen margins (pixels, default: 50, 50, 100, 100)
## Requirements
- Python 3.6+
- NumPy (>=1.21.0)
- SciPy (>=1.7.0)
- Pygame (>=2.0.0)
```bash
pip install -r requirements.txt
```
## Development
Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vibrating-string.git
   ```
## License
This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.
## Author
- Gultekin Yegin (yegingultekin@gmail.com)
