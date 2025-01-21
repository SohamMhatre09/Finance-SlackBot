# AI-Powered Slack Bot

A Slack bot that leverages Gemini AI and Azure services to process PDFs and respond to queries based on your knowledge base.

## Prerequisites

- Python 3.11.9
- Slack Workspace Admin access
- Google Cloud (Gemini API) account
- Azure Document Intelligence Intstance
- ngrok

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SohamMhatre09/Finance-SlackBot
cd Finance-SlackBot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create and configure `.env` file in the root directory:
```plaintext
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_SIGNING_SECRET=your_slack_signing_secret
GEMINI_API_KEY=your_gemini_api_key
AZURE_ENDPOINT=your_azure_endpoint
AZURE_KEY=your_azure_key
```

## Setup nowledge Base

1. Create a `/KnowledgeBase` folder in the root directory if it doesn't exist
2. Add your PDF files to the `/KnowledgeBase` folder
3. Run the extractor to process PDFs:
```bash
python extractor.py
```

## Running the Application

1. Start the Flask application:
```bash
python flask-app.py
```

2. Start ngrok to create a public URL:
```bash
ngrok http 3000
```

## Slack Integration Setup

1. Go to [Slack API](https://api.slack.com/apps) and create a new app
2. Under "OAuth & Permissions", add the following scopes:
   - `chat:write`
   - `app_mentions:read`
   - `channels:history`
   - `groups:history`

3. Install the app to your workspace and copy the Bot User OAuth Token

4. Enable Event Subscriptions:
   - Toggle "Enable Events" to On
   - In the Request URL field, enter your ngrok URL followed by `/slack/events`
   - Example: `https://your-ngrok-url.ngrok.io/slack/events`
   - Subscribe to the `app_mention` event

5. Update your `.env` file with the Slack credentials:
   - Add your Bot User OAuth Token as `SLACK_BOT_TOKEN`
   - Add your Signing Secret as `SLACK_SIGNING_SECRET`

## Usage

1. Invite the bot to your desired Slack channel
2. Mention the bot using @ followed by your query
3. The bot will process your query using the knowledge base and respond accordingly

## Troubleshooting

- Ensure all environment variables are correctly set in `.env`
- Verify ngrok is running and the URL is properly configured in Slack
- Check the application logs for any error messages
- Make sure PDFs are properly processed in the Knowledge Base folder

  ![Example Usage :](https://github.com/user-attachments/assets/0ae08163-e9d4-42b5-b0c8-7f2aedd2ebce)

