# Lex AI - Legal Document Analyzer and Advisor

Lex AI is an AI-powered web application built with Django, HTML, CSS, and JavaScript. It helps users:
- Chat with Terms & Conditions / Privacy Policies
- Ask general legal questions to an AI Legal Assistant
- Upload legal documents and get clause extraction, risk analysis, and obligation summaries.

## Project Structure
```
lexai/
├── chat_app/          # Django app for AI Legal Assistant and Document Analyzer
├── website_chat/      # Django app for Website T&C Chat
├── media/             # Uploaded Documents
├── db.sqlite3         # SQLite database
├── manage.py          # Django management script
├── .env               # API Keys and Secrets (DO NOT COMMIT)
├── requirements.txt   # Python dependencies
├── package.json       # (If any front-end JavaScript package used)
├── static/            # CSS, JavaScript, images
├── templates/         # HTML templates
```

## Setup Instructions

1. Clone the Repository
```bash
git clone <repo-link>
cd lexai
```

2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # (Linux/macOS)
venv\Scripts\activate     # (Windows)
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Setup Environment Variables
Create a `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key_here
DEBUG=True
```

5. Run Migrations
```bash
python manage.py migrate
```

6. Start the Server
```bash
python manage.py runserver
```

7. Access the Application
Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Features
- **Login / Signup Module**
- **Chat with Terms & Conditions**
- **Personal AI Legal Assistant**
- **Document Analyzer (Chat with Documents)**

## Technologies Used
- Backend: Django, Python
- Frontend: HTML5, CSS3, JavaScript
- Database: SQLite3
- AI: Gemini 2.0 API