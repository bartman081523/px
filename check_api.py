from gradio_client import Client
import ssl

# Ignore self-signed certs
client = Client("https://localhost:7860/gradio/", ssl_verify=False)
client.view_api()
