import streamlit as st
import cv2
import numpy as np
import requests

# Function for image preprocessing
def preprocess_image(input_image, target_width, target_height, options):
    if options['use_rgb']:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    if options['use_grayscale']:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    if options['use_blur']:
        input_image = cv2.GaussianBlur(input_image, (options['blur_kernel_size'], options['blur_kernel_size']), 0)
    
    if options['use_sharpen']:
        kernel = np.array([[-1, -1, -1],
                           [-1, options['sharpen_intensity'], -1],
                           [-1, -1, -1]])
        input_image = cv2.filter2D(input_image, -1, kernel)
    
    if options['use_resize']:
        input_image = cv2.resize(input_image, (options['resize_width'], options['resize_height']))
    
    if options['use_morphology']:
        kernel = np.ones((options['morphology_kernel_size'], options['morphology_kernel_size']), np.uint8)
        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,filterSize) 
        if options['use_open']:
            input_image = cv2.morphologyEx(input_image, cv2.MORPH_OPEN, kernel)
        if options['use_close']:
            input_image = cv2.morphologyEx(input_image, cv2.MORPH_CLOSE, kernel)
        if options['use_erode']:
            input_image = cv2.erode(input_image, kernel, iterations=options['erode_iterations'])
        if options['use_dilate']:
            input_image = cv2.dilate(input_image, kernel, iterations=options['dilate_iterations'])
        if options['use_tophat']:
            input_image = cv2.morphologyEx(input_image, cv2.MORPH_TOPHAT, kernel)
        if options['use_blackhat']:
            input_image = cv2.morphologyEx(input_image, cv2.MORPH_BLACKHAT, kernel)

    if options['use_contours']:
        img_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, options['contour_threshold'], 255, cv2.THRESH_BINARY)
        
        mode_mapping = {
            "EXTERNAL": cv2.RETR_EXTERNAL,
            "LIST": cv2.RETR_LIST,
            "CCOMP": cv2.RETR_CCOMP,
            "TREE": cv2.RETR_TREE,
            "FLOODFILL": cv2.RETR_FLOODFILL
        }
        method_mapping = {
            "NONE": cv2.CHAIN_APPROX_NONE,
            "SIMPLE": cv2.CHAIN_APPROX_SIMPLE,
            "APPROX_TC89_L1": cv2.CHAIN_APPROX_TC89_L1,
            "APPROX_TC89_KCOS": cv2.CHAIN_APPROX_TC89_KCOS
        }

        contours, hierarchy = cv2.findContours(thresh, mode=mode_mapping[options['contour_retrieval_mode']], method=method_mapping[options['contour_approx_method']])
        cv2.drawContours(input_image, contours, -1, (0, 255, 0), options['contour_thickness'])
    
    if options['use_canny_edge']:
        # Apply Canny edge detector
        input_image = cv2.Canny(input_image, options['canny_low'], options['canny_high'])

    if options['book_detect']:
        # Apply Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(input_image, 1, np.pi / 180, options['hough_threshold'], minLineLength=options['min_line_length'], maxLineGap=options['max_line_gap'])
        
        # Draw detected lines on the image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(input_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if options['use_cropping']:
        h, w = input_image.shape[:2]
        crop_x = int(w * options['crop_percent'] / 2)
        crop_y = int(h * options['crop_percent'] / 2)
        input_image = input_image[crop_y: h - crop_y, crop_x: w - crop_x]
    
    if options['brightness'] != 0:
        input_image = cv2.convertScaleAbs(input_image, alpha=1, beta=options['brightness'])
    
    if options['contrast'] != 0:
        input_image = cv2.convertScaleAbs(input_image, alpha=options['contrast'])
    
    if options['use_thresholding']:
        _, input_image = cv2.threshold(input_image, options['threshold_value'], 255, cv2.THRESH_BINARY)
    
    if options['use_filtering']:
        kernel = np.array([[-1, -1, -1],
                           [-1, options['filter_intensity'], -1],
                           [-1, -1, -1]])
        input_image = cv2.filter2D(input_image, -1, kernel)
    
    if options['books_detect']:
        api_url = 'https://api.api-ninjas.com/v1/imagetotext/'
        image_file_descriptor = open('D:\CV\preprocessed_image.jpg', 'rb')
        files = {'image': image_file_descriptor}
        api_key = 'nWlwElBzS44OQTb/yJPKXQ==v9V9BlSQ2J9fQ8tw' 
        headers = {'X-Api-Key': api_key}
        r = requests.post(api_url,headers=headers, files=files)
        dic= r.json()
        st.write(r.json())
        pic=""
        for i in range(0,len(dic)):
            if(dic[i]['text']=="-" or dic[i]['text']=="@"):
                pass
            elif('@' in dic[i]['text'] or len(dic[i]['text'])==1):
                pass
            else:
                pic+=dic[i]['text']+" "
        pic= pic + " book"
        st.write("Book Name: ")
        st.write(pic)
        pic = st.text_input("If there is any error in Identification please correct it ", pic)
        if st.button("Search for Books"):
            if pic == " book":
                st.write("No book detected, but you can browse books from these websites:")
            base_url = "https://www.amazon.in"
            search_url = f"{base_url}/s?k={pic.replace(' ', '+')}"
    
            bases_url = "https://www.flipkart.com"
            searches_url = f"{bases_url}/search?q={pic.replace(' ', '%20')}"
    
            basic_url = "https://www.goodreads.com"
            searching_url = f"{basic_url}/search?utf8=%E2%9C%93&query={pic.replace(' ', '+')}"
    
            st.write(search_url)
            st.write(searches_url)
            st.write(searching_url)
    return input_image

st.title("Bookfinder")
st.sidebar.title("Preprocessing Options")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Preprocessing options
options = {
    'use_rgb': st.sidebar.checkbox("RGB"),
    'use_grayscale': st.sidebar.checkbox("Grayscale"),
    'use_blur': st.sidebar.checkbox("Apply Gaussian Blur"),
    'blur_kernel_size': st.sidebar.slider("Blur Kernel Size", 1, 11, 5),
    'use_sharpen': st.sidebar.checkbox("Sharpen"),
    'sharpen_intensity': st.sidebar.slider("Sharpen Intensity", 1, 10, 5),
    'use_resize': st.sidebar.checkbox("Resize"),
    'resize_width': st.sidebar.slider("Resize Width", 100, 2000, 800),
    'resize_height': st.sidebar.slider("Resize Height", 100, 2000, 600),
    'use_morphology': st.sidebar.checkbox("Morphological Operations"),
    'morphology_kernel_size': st.sidebar.slider("Morphology Kernel Size", 1, 11, 5),
    'use_open': st.sidebar.checkbox("Use Open Operation"),
    'use_close': st.sidebar.checkbox("Use Close Operation"),
    'use_tophat': st.sidebar.checkbox("Use Top Hat Operation"),
    'use_blackhat': st.sidebar.checkbox("Use Black Hat Operation"),
    'use_erode': st.sidebar.checkbox("Use Erode Operation"),
    'erode_iterations': st.sidebar.slider("Erode Iterations", 1, 10, 1),
    'use_dilate': st.sidebar.checkbox("Use Dilate Operation"),
    'dilate_iterations': st.sidebar.slider("Dilate Iterations", 1, 10, 1),
    'use_contours': st.sidebar.checkbox("Contour Detection"),
    'contour_retrieval_mode': st.sidebar.selectbox("Contour Retrieval Mode", ["EXTERNAL", "LIST", "CCOMP", "TREE", "FLOODFILL"]),
    'contour_approx_method': st.sidebar.selectbox("Contour Approximation Method", ["NONE", "SIMPLE", "APPROX_TC89_L1", "APPROX_TC89_KCOS"]),
    'contour_thickness': st.sidebar.slider("Contour Thickness", 1, 5, 2),
    'contour_threshold': st.sidebar.slider("Contour Threshold", 0, 255, 128),
    'use_cropping': st.sidebar.checkbox("Crop Center"),
    'crop_percent': st.sidebar.slider("Crop Percentage", 1, 50, 25),
    'brightness': st.sidebar.slider("Brightness", -100, 100, 0),
    'contrast': st.sidebar.slider("Contrast", -100, 100, 0),
    'use_thresholding': st.sidebar.checkbox("Thresholding"),
    'threshold_value': st.sidebar.slider("Threshold Value", 0, 255, 128),
    'use_filtering': st.sidebar.checkbox("Filtering"),
    'filter_intensity': st.sidebar.slider("Filter Intensity", 1, 10, 5),
    'use_canny_edge': st.sidebar.checkbox("Canny Edge Detection"),
    'canny_low': st.sidebar.slider("Canny Low Threshold", 0, 255, 50),
    'canny_high': st.sidebar.slider("Canny High Threshold", 0, 255, 150),
    'book_detect': st.sidebar.checkbox("Hough Transform"),
    'hough_threshold': st.sidebar.slider("Hough Threshold", 1, 500, 100),
    'min_line_length': st.sidebar.slider("Min Line Length", 1, 500, 100),
    'max_line_gap': st.sidebar.slider("Max Line Gap", 1, 500, 10),
    'books_detect': st.sidebar.checkbox("Detect Book")
}

if uploaded_image is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)

    # Preprocess the image based on selected options
    preprocessed_image = preprocess_image(image, 800, 600, options)

    # Display the original and preprocessed images
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    st.subheader("Preprocessed Image")
    st.image(preprocessed_image, use_column_width=True)

    save_path = 'D:\CV\preprocessed_image.jpg'
    cv2.imwrite(save_path, preprocessed_image)

    print(options['books_detect'])

    

    

    
