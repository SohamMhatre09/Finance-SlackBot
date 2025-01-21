# Slack Bot with Gemini and Azure Integration

This project integrates a Slack bot with the Gemini API and Azure services to provide intelligent responses and document processing. The bot can process PDFs from a knowledge base, respond to user queries, and interact with Slack channels. Follow the steps below to set up and run the application.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup Instructions](#setup-instructions)
   - [Install Dependencies](#1-install-dependencies)
   - [Configure Environment Variables](#2-configure-environment-variables)
   - [Add PDFs to Knowledge Base](#3-add-pdfs-to-knowledge-base)
   - [Run the Application](#4-run-the-application)
   - [Configure Slack App](#5-configure-slack-app)
3. [Running the Bot](#running-the-bot)
4. [Folder Structure](#folder-structure)
5. [Notes](#notes)

---

## Prerequisites

Before starting, ensure you have the following:

1. **Python 3.8+**: Install Python from [python.org](https://www.python.org/).
2. **Ngrok**: Required for exposing your local server to the internet. Download it from [ngrok.com](https://ngrok.com/).
3. **Slack App**: A registered Slack app with the necessary permissions.
4. **API Keys**:
   - Gemini API key (from [Gemini](https://gemini.com/)).
   - Azure endpoint and key (from [Azure Portal](https://portal.azure.com/)).

---

## Setup Instructions

### 1. Install Dependencies

Install the required Python packages by running:

```bash
pip install -r requirements.txt
2. Configure Environment Variables
Create a .env file in the root directory and add the following variables:

plaintext
Copy
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_SIGNING_SECRET=your_slack_signing_secret
GEMINI_API_KEY=your_gemini_api_key
AZURE_ENDPOINT=your_azure_endpoint
AZURE_KEY=your_azure_key
Replace the placeholders with your actual credentials.

3. Add PDFs to Knowledge Base
Place your PDF files in the /KnowledgeBase folder. These files will be processed by the application to extract relevant information.

4. Run the Application
Extract Data from PDFs:
Run the extractor.py script to process the PDFs in the /KnowledgeBase folder:

bash
Copy
python extractor.py
Start the Flask App:
Run the Flask application:

bash
Copy
python flask-app.py
Expose the App with Ngrok:
Use Ngrok to forward the local server to a public URL:

bash
Copy
ngrok http 3000
Copy the forwarding URL provided by Ngrok (e.g., https://<ngrok-url>).

5. Configure Slack App
Register a Slack App:

Go to the Slack API website.

Click "Create New App" and choose "From scratch."

Name your app and select the workspace where it will be installed.

Configure Permissions:

Under "OAuth & Permissions," add the following scopes:

chat:write (to send messages as the bot).

app_mentions:read (to listen for messages where the bot is mentioned).

channels:history or groups:history (to read messages in channels or groups).

Install the app to your workspace and note the Bot User OAuth Token.

Set Up Event Subscriptions:

Enable "Event Subscriptions" and provide the Request URL:

Copy
https://<ngrok-url>/slack/events
Subscribe to the app_mention event so the bot can respond when mentioned.

Install Slack SDK:

Use the slack_bolt Python SDK to simplify Slack app development.

Running the Bot
Start the Flask app and Ngrok as described above.

Mention the bot in your Slack workspace to interact with it.

Folder Structure
Copy
.
├── .env                  # Environment variables
├── KnowledgeBase/        # Folder for PDF files
├── extractor.py          # Script to process PDFs
├── flask-app.py          # Flask application
├── requirements.txt      # Python dependencies
└── README.md             # This file
Notes
Ensure your Ngrok URL is updated in the Slack app settings whenever you restart Ngrok.

Replace placeholders in the .env file with your actual API keys and tokens.

For local development, keep the Flask app and Ngrok running simultaneously.
