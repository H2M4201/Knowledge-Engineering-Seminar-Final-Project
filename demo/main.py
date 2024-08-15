import subprocess
from pyngrok import ngrok

# Start Streamlit app in the background
streamlit_process = subprocess.Popen(["streamlit", "run", "demo.py"])

# Connect ngrok to the same port as your Streamlit app

ngrok.set_auth_token("2jB9aNcOBVL5UiJpMaKylQqWZ5d_7TLre1fAaMEJ3HPd6w2MU")
public_url = ngrok.connect(8501)

# Print the public URL
print("Public URL:", public_url)

# Wait for user to terminate
input("Press Enter to stop...")

# Terminate Streamlit app and ngrok
streamlit_process.terminate()
ngrok.kill()
