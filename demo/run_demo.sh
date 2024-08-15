# Description: Run the demo.py file and expose it to the internet using ngrok

# remember to run `ngrok authtoken $NGROK_TOKEN` before running this script

ngrok http 8501 &
streamlit run demo.py

