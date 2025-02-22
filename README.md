# SIGN-O-VOICE

**SIGN-O-VOICE** is an innovative application designed to bridge the communication gap between the Deaf and Hard of Hearing (DHH) community and others. By translating sign language gestures into coherent English sentences, it enables real-time, effortless interactions.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Features

- **Real-time Translation**: Converts sign language gestures into English sentences instantly.
- **User-Friendly Interface**: Simple and intuitive design for ease of use.
- **Extensive Vocabulary**: Supports a wide range of gestures for comprehensive communication.

## Installation

To set up **SIGN-O-VOICE**, ensure you have Python 3.9 installed. Follow the steps below:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/HotPolarbear/MP16-SIGNO-VOICE.git
   cd sign-o-voice
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python3.9 -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. **Install Dependencies**:
   Install the required libraries with compatible versions:
   ```bash
   pip install tensorflow==2.12.0
   pip install mediapipe==0.10.0
   pip install numpy==1.23.5
   pip install matplotlib==3.8.2
   pip install scikit-learn==1.2.2
   ```

   **Library Explanations**:
   - **TensorFlow 2.12.0**: An open-source machine learning library used for training and deploying models. [TensorFlow Installation Guide](https://www.tensorflow.org/install/pip)
   - **MediaPipe 0.10.0**: A framework for building multimodal (e.g., video, audio) applied machine learning pipelines, utilized here for hand gesture recognition.
   - **NumPy 1.23.5**: A fundamental package for numerical computations in Python.
   - **Matplotlib 3.8.2**: A plotting library for creating static, animated, and interactive visualizations.
   - **Scikit-learn 1.2.2**: A machine learning library featuring various algorithms and tools for data analysis.

   *Note*: These versions are selected for compatibility with Python 3.9 and to ensure seamless integration.

## Usage

After installation, you can start the application with:

```bash
python main.py
```

**Using SIGN-O-VOICE**:

1. **Launch the Application**: Run the above command to start.
2. **Position the Camera**: Ensure your webcam is properly positioned to capture your hand gestures.
3. **Perform Gestures**: Begin signing; the application will translate recognized gestures into English sentences displayed on the screen.

*For detailed instructions and troubleshooting, refer to the [User Manual](docs/user_manual.md).*

## Contributing

We welcome contributions to enhance **SIGN-O-VOICE**. To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a pull request.

Please ensure your code adheres to our coding standards and includes relevant tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed by **Shubham Bhatt**, **Prashant Kumar Singh**, **Diya Bhatia**, and **Vanshika Goyal** under the guidance of **Mr. Rajeev Kumar Singh** at **KIET Group of Institutions**.

## Contact

For inquiries or support, please contact:
- **Shubham Bhatt**
  - Email: [shubham.2125csme1045@kiet.edu](mailto:shubham.2125csme1045@kiet.edu)
  - LinkedIn: [linkedin.com/in/shubham-bhatt-ai-n-design](https://www.linkedin.com/in/shubham-bhatt-ai-n-design)

- **Prashant Kumar Singh**
  - Email: [prashant.2125csme1@kiet.edu](mailto:prashant.2125csme1@kiet.edu)
  - LinkedIn: [linkedin.com/in/prashant70](https://www.linkedin.com/in/prashant70)

---

