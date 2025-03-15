# Ollama Llama Vision Demo on Google Colab

## Overview

This repository demonstrates an end-to-end demo that leverages the advanced Llama Vision model via the Ollama API within a Google Colab notebook. The demo enables you to:

- **Upload and Display Images:** Seamlessly upload images (JPG, PNG, WEBP, or TIFF) and view them directly in the notebook.
- **Interactive Analysis:** Replace previous images upon new uploads and enter custom prompts to tailor the image analysis.
- **Cloud Compute:** Utilize Google Colab's free compute resources for running AI models, making it accessible and efficient.

## About Llama Vision

Llama Vision is a state-of-the-art computer vision model that integrates image understanding with language generation. It is designed to:
- **Generate Detailed Descriptions:** Provide comprehensive narratives about the content of an image.
- **Identify Objects and Scenes:** Recognize various objects and analyze the context of scenes.
- **Answer Visual Questions:** Respond to user queries regarding the visual content.

This model’s unique combination of vision and language capabilities makes it ideal for applications in content analysis, interactive data exploration, and more.

## Advantages of Using Ollama

Ollama offers a user-friendly interface to interact with sophisticated AI models like Llama Vision. Key advantages include:

- **Simplicity:** Easily integrate and communicate with AI models using the `ollama` Python library.
- **Flexibility:** Quickly switch between different models and configurations without complex setup.
- **Efficiency:** Streamline your workflow with straightforward API calls that reduce overhead and accelerate development.
- **Interactive Workflows:** Designed to work seamlessly in environments like Google Colab, supporting rapid prototyping and collaborative projects.

## Benefits of Using Google Colab

Google Colab provides a powerful, cloud-based platform ideal for AI experimentation and prototyping:

- **Free Compute Resources:** Access GPUs and TPUs without any cost, accelerating your model experiments.
- **Ease of Use:** Enjoy a fully configured environment that supports interactive notebooks, reducing the setup time.
- **Collaboration:** Share your notebooks effortlessly with colleagues or the community for enhanced collaboration.
- **Rapid Prototyping:** Experiment and iterate quickly in an interactive and accessible platform.

## Demo Workflow

In this demo, the process is streamlined into a few simple steps:
1. **Image Upload:** Use an intuitive widget to upload your image. The system supports various formats and automatically clears the previous image upon a new upload.
2. **Image Display:** Immediately view the uploaded image within the notebook for confirmation.
3. **Custom Prompt Input:** Enter a custom prompt (with a helpful default provided) to specify the analysis requirements.
4. **Image Analysis:** The image, along with your prompt, is sent to the Llama Vision model through the Ollama API, and the analysis results are displayed in the notebook.

## Getting Started

To run the demo:
- Open the provided Colab notebook.
- Follow the setup instructions to install the necessary packages.
- Execute the cells in sequence to upload an image, input your custom prompt, and view the detailed analysis provided by Llama Vision.

```mermaid
flowchart TD
    A[Start Colab Notebook]
    B[Upload Image using FileUpload Widget]
    C[Image Upload Change Event Triggered]
    D[Clear previous image display]
    E[Extract uploaded image content]
    F[Display image in Image Output area]
    G[Reset FileUpload widget]
    H[Enter custom prompt default: What is in this image?]
    I[Click Analyze Image Button]
    J[Analyze Button Click Event Triggered]
    K[Clear previous analysis output]
    L{Is image uploaded?}
    M[Save image to disk uploaded_image.jpeg]
    N[Retrieve prompt text from Text Widget]
    O[Call Ollama.chat API with image and prompt]
    P[Display API response in Analysis Output]
    Q[End Process]
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L -- Yes --> M
    L -- No --> P[Display error: Please upload an image before analyzing]
    M --> N
    N --> O
    O --> P
    P --> Q
  ```

## Conclusion

This demo showcases the powerful integration of advanced AI models like Llama Vision with the simplicity of the Ollama API, all within the scalable and collaborative environment of Google Colab. Whether you’re exploring AI for research, development, or educational purposes, this demo provides a hands-on experience with cutting-edge image analysis tools.

## License

This project is licensed under the [MIT License](LICENSE).

Feel free to contribute, report issues, or share your feedback!
