
# Image Processing Tool

An interactive web application built with **Streamlit** and **OpenCV** for performing various image processing operations. The app allows users to upload an image and apply transformations such as background removal, edge detection, blurring, resizing, rotation, and image addition.
## 🔗 Live Demo
Check out the live app here: [Image Processing Tool](https://digital-image-processing-lab-project-gjbtpv6ldcf4sjl9gvobe4.streamlit.app/)

## Features

- **Background Removal**: Automatically removes the background from images.
- **Edge Detection**: Detect edges using the Canny edge detection algorithm.
- **Image Blurring**: Apply Gaussian blur with adjustable intensity.
- **Image Resizing**: Resize images to specified dimensions.
- **Image Rotation**: Rotate images to a custom angle.
- **Image Addition**: Blend two images with equal weights.

## Installation

### Prerequisites
- Python 3.8 or above
- Virtual environment (optional but recommended)

### Steps
1. Clone this repository or download the source code.
2. Install the required libraries using pip:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Steps to Run the Application
1. Activate the virtual environment:
   ```bash
   .\myenv\Scripts\activate
   ```
2. Navigate to the project directory:
   ```bash
   cd C:\Users\arier\OneDrive\Desktop
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```


## File Structure
- `app.py`: Main application script.
- `requirements.txt`: Contains a list of all required Python libraries.
- `report.pdf` (or similar): A report file describing the app functionality and implementation details.



## Dependencies

The following Python libraries are used:
- **Streamlit**: For building the web interface.
- **OpenCV**: For image processing operations.
- **NumPy**: For array and matrix operations.
- **Pillow**: For additional image manipulations.

Install these using:
```bash
pip install streamlit opencv-python-headless numpy pillow
```


## How to Use the App
1. Upload an image in JPG, PNG, BMP, or JPEG format.
2. Select the desired operations from the sidebar:
   - Adjust the parameters (e.g., blur scale, rotation angle) as needed.
3. View the processed image.
4. Download the processed image using the "Download Processed Image" button.

