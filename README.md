# fastapi-huggingface
FastAPI backend integrated with Hugging Face models for AI-powered APIs

## Features

- **FastAPI Integration**: Leverage the power of FastAPI for building high-performance APIs.
- **Hugging Face Models**: Seamlessly integrate state-of-the-art machine learning models from Hugging Face.
- **Scalable and Modular**: Designed for scalability and easy customization.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/rpankaj853/fastapi-huggingface.git
    cd fastapi-huggingface
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the FastAPI server:
    ```bash
    uvicorn app.main:app --reload
    ```

2. Access the API documentation at `http://127.0.0.1:8000/docs`.

3. Set up your Hugging Face access token:
    - Obtain your token from [Hugging Face](https://huggingface.co/settings/tokens).
    - Export it as an environment variable:
      ```bash
      export HF_ACCESS_TOKEN=your_huggingface_token
      ```

4. (Optional) Use your custom service token for additional authentication:
    - Define your custom token:
      ```bash
      export CUSTOM_SERVICE_TOKEN=your_custom_service_token
      ```

## Example Endpoints

- **Text Classification**: Send a POST request with text input to classify it.
- **Text Generation**: Generate text using pre-trained Hugging Face models.
- **Custom Models**: Easily integrate your own Hugging Face models.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
