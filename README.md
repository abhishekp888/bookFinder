# bookFinder
# BookFinder Project

**BookFinder** is an innovative system developed using OpenCV and Streamlit for the automatic extraction and summarization of essential information from images of book covers. The primary goal is to streamline the process of obtaining critical details such as book title, author, and publication information from book cover images, offering users a quick and convenient way to access and share concise summaries.

## Project Highlights

### 1. Image Preprocessing
- Utilizes OpenCV for image processing to enhance the quality and readability of book cover images.
- Various preprocessing options provided, including RGB conversion, grayscale, Gaussian blur, resizing, morphological operations, contour detection, and more.

### 2. Text Extraction
- Implements Optical Character Recognition (OCR) techniques to accurately extract textual information from the book cover, including title, author name, and publication details.

### 3. Visual Feature Extraction
- Utilizes OpenCV for the identification and extraction of relevant visual features from book cover images, potentially including logos, images, or patterns.

### 4. Information Summarization
- Organizes the extracted information into a structured and coherent summary, presenting key details in a user-friendly format.

### 5. User Interface (UI)
- Developed an intuitive Streamlit-based user interface allowing users to easily upload book cover images, view extracted information, and access summarized details.
- Various preprocessing options can be selected through the Streamlit sidebar for customization.

### 6. Deployment
To deploy the BookFinder project using Streamlit, follow these steps:

1. Ensure you have Streamlit installed:
    ```bash
    pip install streamlit
    ```

2. Clone the repository:
    ```bash
    git clone https://github.com/your-username/BookFinder.git
    ```

3. Navigate to the project directory:
    ```bash
    cd BookFinder
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run bookfinder_app.py
    ```

5. Open the provided URL in your browser.

## Access BookFinder App
Explore BookFinder by accessing the deployed Streamlit app at [https://bookfinder-pttiukkhnqylhefmu5vncy.streamlit.app/](https://bookfinder-pttiukkhnqylhefmu5vncy.streamlit.app/).

Feel free to upload book cover images, customize preprocessing options, and experience the automatic extraction and summarization of essential book information.

