# Potato Leaf Classifier

This is a potato leaf image classifier that analyzes an image of a potato leaf and verifies if it has blight on its leaf.

## Classification

When you input the image, the model will return three possible labels:

- **(0) Healthy**: Healthy leaf
- **(1) Early Blight**: Early blight
- **(2) Late Blight**: Late blight

## How to Use

1. Upload the potato leaf image you want to classify in the appropriate section.
2. The model will analyze the image and return the corresponding label.

## Example

Send your image below to scan and return the result.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/potato-leaf-classifier.git
    cd potato-leaf-classifier
    ```

2. Create and activate a virtual environment (optional, but recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # For Windows: env\Scripts\activate
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    streamlit run app.py
    ```

## Dependencies

- Python 3.7 or higher
- Streamlit
- Matplotlib
- Numpy
- Pandas

## Project Structure

```plaintext
potato-leaf-classifier/
│
├── app.py               # Main Streamlit application file
├── requirements.txt     # Project dependencies list
├── README.md            # Project documentation
└── models/              # Directory containing the trained model (if applicable)
```
## Links
Streamlit: https://potatoclassifier.streamlit.app/
