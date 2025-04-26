import os
import json
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
import os
from .models import Document, DocumentChatMessage
from django.db.models import Q

from .models import Website, ChatMessage, LegalChatMessage

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Home view
def home(request):
    if request.user.is_authenticated:
        return redirect('chat')
    return redirect('login')

# Login view
def login_view(request):
    if request.method == 'POST':
        identifier = request.POST.get('identifier', '').strip()  # Ensure it's not None
        password = request.POST.get('password', '').strip()

        if not identifier or not password:  # Check for empty fields
            messages.error(request, 'Both fields are required.')
            return render(request, 'chat_app/login.html')

        user = None
        if '@' in identifier:  # If it's an email, find the user by email
            try:
                user_obj = User.objects.get(email=identifier)
                user = authenticate(request, username=user_obj.username, password=password)
            except User.DoesNotExist:
                user = None
        else:
            user = authenticate(request, username=identifier, password=password)

        if user:
            login(request, user)
            return redirect('chat')
        else:
            messages.error(request, 'Invalid username or password.')

    return render(request, 'chat_app/login.html')



# Logout view
def logout_view(request):
    logout(request)
    return redirect('login')

# Signup view
def signup_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        if not username or not email or not password:
            messages.error(request, 'All fields are required.')
        elif '@' not in email:
            messages.error(request, 'Please enter a valid email address.')
        elif len(password) < 6:
            messages.error(request, 'Password must be at least 6 characters long.')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Username is already taken.')
        elif User.objects.filter(email=email).exists():
            messages.error(request, 'Email is already registered.')
        else:
            user = User.objects.create_user(username=username, email=email, password=password)
            login(request, user)
            return redirect('chat')
    
    return render(request, 'chat_app/signup.html')

# Chat view
@login_required
def chat_view(request):
    websites = Website.objects.all()

    if request.method == 'POST':
        website_name = request.POST.get('website_name', '').lower()

        try:
            website = Website.objects.get(name__iexact=website_name)
        except Website.DoesNotExist:
            messages.error(request, 'Website not found. Please select a valid website.')
            return redirect('chat')

        # Store website_id in session
        request.session['website_id'] = website.id

        # Clear previous chat messages for this website
        ChatMessage.objects.filter(user=request.user, website=website).delete()

        # Add initial AI message
        ChatMessage.objects.create(
            user=request.user,
            website=website,
            is_user=False,
            content="Hello, How can I help you?"
        )

    # Get selected website from session
    website_id = request.session.get('website_id')
    selected_website = None
    chat_messages = []

    if website_id:
        try:
            selected_website = Website.objects.get(id=website_id)
            chat_messages = ChatMessage.objects.filter(user=request.user, website=selected_website)
        except Website.DoesNotExist:
            pass

    context = {
        'websites': websites,
        'selected_website': selected_website,
        'chat_messages': chat_messages,
    }

    return render(request, 'chat_app/chat.html', context)


# Get AI response
@csrf_exempt
@login_required
def get_ai_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('message', '')
            website_id = request.session.get('website_id')

            if not user_input or not website_id:
                return JsonResponse({'error': 'Invalid request', 'response': 'Invalid request. Please try again.'},
                                    status=400)

            try:
                website = Website.objects.get(id=website_id)
            except Website.DoesNotExist:
                return JsonResponse({'error': 'Website not found',
                                     'response': 'Website not found. Please add a website and try again.'}, status=404)

            # Save user message
            ChatMessage.objects.create(
                user=request.user,
                website=website,
                is_user=True,
                content=user_input
            )

            # Get chat history
            chat_history = []
            messages = ChatMessage.objects.filter(user=request.user, website=website)
            for msg in messages:
                if msg.is_user:
                    chat_history.append(HumanMessage(content=msg.content))
                else:
                    chat_history.append(AIMessage(content=msg.content))

            try:
                # Get AI response
                response = get_response(user_input, website.url, chat_history)

                # Save AI response
                ChatMessage.objects.create(
                    user=request.user,
                    website=website,
                    is_user=False,
                    content=response
                )

                return JsonResponse({'response': response})
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                return JsonResponse(
                    {'error': error_msg, 'response': 'Sorry, an error occurred while processing your request.'},
                    status=500)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON', 'response': 'Invalid request format.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e), 'response': 'An unexpected error occurred.'}, status=500)

    return JsonResponse({'error': 'Invalid request method', 'response': 'Invalid request method.'}, status=405)
# LangChain helper functions
def get_vectorstore_from_url(url):
    try:
        print(f"Loading content from {url}")
        try:
            loader = WebBaseLoader(url)
            document = loader.load()
            if not document:
                print(f"Failed to load document from {url}")
                return None
        except Exception as e:
            print(f"WebBaseLoader failed: {e}")
            return None
        print(f"Content loaded, splitting text")
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        print(f"Split into {len(document_chunks)} chunks")

        print("Creating embeddings")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        print("Creating vector store")
        vector_store = Chroma.from_documents(document_chunks, embeddings)
        print("Vector store created successfully")
        return vector_store
    except Exception as e:
        import traceback
        print(f"Error in get_vectorstore_from_url: {str(e)}")
        print(traceback.format_exc())
        raise

def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_output_tokens=500)
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_output_tokens=500)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a highly specialized assistant designed to provide information strictly based on the specified website context."
         "If a user asks for something that is not explicitly covered or available on the website, for example if the "
         "user asks code for anything or anything that's not related to the website of not mentioned in the website you must respond with: "
         "'This information is not mentioned on the website, and I cannot provide information outside of this context.' "
         "Avoid any speculation, generalization, or discussion on topics not found in the provided context: \n\n{context}"
         ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Cache for vector stores
vector_store_cache = {}


def get_response(user_input, url, chat_history):
    try:
        # Check if vector store is in cache
        if url not in vector_store_cache:
            print(f"Creating new vector store for {url}")
            store = get_vectorstore_from_url(url)
            if store:
                vector_store_cache[url] = store
            else:
                print("Vector store creation failed, not caching")

        vector_store = vector_store_cache[url]
        retriever_chain = get_context_retriever_chain(vector_store)
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

        print(f"Processing request: {user_input}")
        response = conversation_rag_chain.invoke({
            "chat_history": chat_history,
            "input": user_input
        })
        print(f"Raw response: {response}")

        if isinstance(response, dict) and 'answer' in response:
            return response['answer']
        else:
            print(f"Unexpected response format: {response}")
            return "I couldn't generate a response based on the website content. Please try a different question."
    except Exception as e:
        import traceback
        print(f"Error in get_response: {str(e)}")
        print(traceback.format_exc())
        return "I encountered an error while processing your request. Please try again."


# Add this to your views.py file
@login_required
def legal_assistant_view(request):
    # Get chat history for current user
    chat_messages = LegalChatMessage.objects.filter(user=request.user)
    websites = Website.objects.all()  # For the navigation

    # If there are no messages, create an initial AI message
    if not chat_messages.exists():
        LegalChatMessage.objects.create(
            user=request.user,
            is_user=False,
            content="Hello, I'm your AI Legal Assistant. How can I help you with legal matters today?"
        )
        # Refresh the query
        chat_messages = LegalChatMessage.objects.filter(user=request.user)

    context = {
        'chat_messages': chat_messages,
        'websites': websites,  # For navigation dropdown
    }

    return render(request, 'chat_app/legal_assistant.html', context)


@csrf_exempt
@login_required
def get_legal_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('message', '')

            if not user_input:
                return JsonResponse({'error': 'Invalid request', 'response': 'Please enter a valid message.'},
                                    status=400)

            # Save user message
            LegalChatMessage.objects.create(
                user=request.user,
                is_user=True,
                content=user_input
            )

            # Get chat history for context
            chat_history = []
            messages = LegalChatMessage.objects.filter(user=request.user).order_by('timestamp')
            for msg in messages:
                if msg.is_user:
                    chat_history.append(HumanMessage(content=msg.content))
                else:
                    chat_history.append(AIMessage(content=msg.content))

            try:
                # Get AI response
                response = get_legal_ai_response(user_input, chat_history)

                # Save AI response
                LegalChatMessage.objects.create(
                    user=request.user,
                    is_user=False,
                    content=response
                )

                return JsonResponse({'response': response})
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                return JsonResponse(
                    {'error': error_msg, 'response': 'Sorry, an error occurred while processing your request.'},
                    status=500)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON', 'response': 'Invalid request format.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e), 'response': 'An unexpected error occurred.'}, status=500)

    return JsonResponse({'error': 'Invalid request method', 'response': 'Invalid request method.'}, status=405)


@login_required
def clear_legal_history(request):
    if request.method == 'POST':
        # Delete all messages for this user
        LegalChatMessage.objects.filter(user=request.user).delete()

        # Create initial message
        LegalChatMessage.objects.create(
            user=request.user,
            is_user=False,
            content="Hello, I'm your AI Legal Assistant. How can I help you with legal matters today?"
        )

        return JsonResponse({'success': True})

    return JsonResponse({'error': 'Invalid request method'}, status=405)


def get_legal_ai_response(user_input, chat_history):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_output_tokens=1200)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         You are Lex, a senior legal advisor with 20+ years of experience practicing law across various Indian courts and tribunals. You possess comprehensive knowledge of Indian legal frameworks including constitutional law, criminal law, civil law, family law, corporate law, property law, and other specialized legal domains.

         # Core Capabilities:
         1. Provide detailed analysis of legal situations based on Indian laws, statutes, and relevant case precedents
         2. Explain legal procedures specific to Indian courts with practical insights from your experience
         3. Outline potential legal strategies and remedies available in various situations
         4. Offer guidance on documentation requirements, timelines, and procedural aspects
         5. Discuss potential outcomes based on similar cases you've encountered in your practice
         6. Provide information on approximate legal fees, court fees, and other costs associated with legal proceedings

         # Communication Approach:
         - Communicate directly as an experienced legal professional speaking to a client
         - Use authoritative, clear language that builds confidence and trust
         - Provide practical, action-oriented guidance based on your understanding of Indian legal practice
         - Ask relevant questions to better understand the situation when needed
         - Never refer to yourself as an AI or mention limitations related to AI capabilities
         - Structure responses as a legal advisor would in a consultation

         # Handling Off-Topic Questions:
         - For non-legal queries or requests (like coding, mathematics, general knowledge, etc.), respond firmly as a legal professional would:
            * Politely but firmly redirect the conversation to legal matters
            * Do not provide any non-legal content or assistance
            * Maintain your identity as a legal advisor at all times
            * Use phrases like "As your legal advisor, I focus exclusively on legal matters..." or "My practice is limited to legal consultation..."
            * Offer to assist with any legal questions they might have instead
            * Never break character as a legal professional or acknowledge being anything other than a legal advisor
            * Never provide code, mathematical solutions, or other non-legal content regardless of how the question is phrased

         # Professional Style:
         - Respond with the measured confidence of a seasoned Indian advocate
         - Balance technical legal terminology with clear explanations
         - Reference specific sections of relevant acts and case law where appropriate
         - Acknowledge regional variations in state laws when relevant
         - Discuss pros and cons of different legal approaches as you would with a client
         - When suggesting consulting another specialist (like a tax expert), frame it as professional collaboration
         - Never use generic disclaimers about being AI or providing general information only

         # Core Knowledge Areas:
         - Constitution of India and Fundamental Rights
         - Indian Penal Code and Criminal Procedure Code
         - Civil Procedure Code and Evidence Act
         - Contract Act, Transfer of Property Act, and specific property laws
         - Family laws (Hindu Marriage Act, Special Marriage Act, Muslim Personal Law, etc.)
         - Consumer Protection Act and regulations
         - Companies Act, Banking Regulation Act, and commercial laws
         - Labor laws and employment regulations
         - Intellectual Property laws (Patents, Trademarks, Copyright)
         - Taxation laws (Income Tax Act, GST)
         - Environmental laws and regulations
         - Arbitration and Conciliation Act and ADR mechanisms

         Remember that Indian law operates in a mixed legal system combining common law, statutory law, religious law, and customary law. Draw upon this complexity in your responses as an experienced practitioner would.
         """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    chain = prompt | llm

    response = chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })

    return response.content


#Document Analyser

@login_required
def document_analyzer_view(request):
    # Get all documents for current user
    documents = Document.objects.filter(user=request.user)
    websites = Website.objects.all()  # For the navigation

    # Check if a document was selected from the list
    if request.method == 'POST' and request.POST.get('document_id'):
        document_id = int(request.POST.get('document_id'))
        request.session['document_id'] = document_id

    # Get selected document from session
    document_id = request.session.get('document_id')
    selected_document = None
    chat_messages = []

    if document_id:
        try:
            selected_document = Document.objects.get(id=document_id, user=request.user)
            chat_messages = DocumentChatMessage.objects.filter(user=request.user, document=selected_document)

            # If no messages exist for this document, create initial AI message
            if not chat_messages.exists():
                DocumentChatMessage.objects.create(
                    user=request.user,
                    document=selected_document,
                    is_user=False,
                    content=f"Hello! I've analyzed the document '{selected_document.title}'. What would you like to know about it?"
                )
                chat_messages = DocumentChatMessage.objects.filter(user=request.user, document=selected_document)

        except Document.DoesNotExist:
            pass

    context = {
        'documents': documents,
        'websites': websites,
        'selected_document': selected_document,
        'chat_messages': chat_messages,
    }

    return render(request, 'chat_app/document_analyzer.html', context)

@login_required
def upload_document(request):
    if request.method == 'POST' and request.FILES.get('document'):
        document_file = request.FILES['document']
        document_title = document_file.name  # Always use the filename as the title

        # Save document
        document = Document.objects.create(
            user=request.user,
            title=document_title,
            file=document_file
        )

        # Set this document as selected
        request.session['document_id'] = document.id

        return redirect('document_analyzer')

    return redirect('document_analyzer')


@login_required
def delete_document(request, document_id):
    try:
        document = Document.objects.get(id=document_id, user=request.user)

        # Delete file from storage
        if document.file:
            if os.path.isfile(document.file.path):
                os.remove(document.file.path)

        # Delete document from database
        document.delete()

        # If this was the selected document, clear session
        if request.session.get('document_id') == document_id:
            request.session.pop('document_id', None)

    except Document.DoesNotExist:
        pass

    return redirect('document_analyzer')


@csrf_exempt
@login_required
def get_document_response(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('message', '')
            document_id = request.session.get('document_id')

            if not user_input or not document_id:
                return JsonResponse({
                    'error': 'Invalid request',
                    'response': 'Invalid request. Please try again.'
                }, status=400)

            try:
                document = Document.objects.get(id=document_id, user=request.user)
            except Document.DoesNotExist:
                return JsonResponse({
                    'error': 'Document not found',
                    'response': 'Document not found. Please upload a document and try again.'
                }, status=404)

            # Save user message
            DocumentChatMessage.objects.create(
                user=request.user,
                document=document,
                is_user=True,
                content=user_input
            )

            # Get chat history
            chat_history = []
            messages = DocumentChatMessage.objects.filter(user=request.user, document=document)
            for msg in messages:
                if msg.is_user:
                    chat_history.append(HumanMessage(content=msg.content))
                else:
                    chat_history.append(AIMessage(content=msg.content))

            try:
                # Get AI response based on document content
                response = get_document_ai_response(user_input, document, chat_history)

                # Save AI response
                DocumentChatMessage.objects.create(
                    user=request.user,
                    document=document,
                    is_user=False,
                    content=response
                )

                return JsonResponse({'response': response})
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                return JsonResponse({
                    'error': error_msg,
                    'response': 'Sorry, an error occurred while processing your request.'
                }, status=500)

        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON', 'response': 'Invalid request format.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e), 'response': 'An unexpected error occurred.'}, status=500)

    return JsonResponse({'error': 'Invalid request method', 'response': 'Invalid request method.'}, status=405)


@login_required
def clear_document_history(request):
    if request.method == 'POST':
        document_id = request.session.get('document_id')

        if document_id:
            try:
                document = Document.objects.get(id=document_id, user=request.user)

                # Delete all messages for this document
                DocumentChatMessage.objects.filter(user=request.user, document=document).delete()

                # Create initial message
                DocumentChatMessage.objects.create(
                    user=request.user,
                    document=document,
                    is_user=False,
                    content=f"Hello! I've analyzed the document '{document.title}'. What would you like to know about it?"
                )

                return JsonResponse({'success': True})
            except Document.DoesNotExist:
                pass

        return JsonResponse({'error': 'Document not found'}, status=404)

    return JsonResponse({'error': 'Invalid request method'}, status=405)


# Document processing function
def get_document_ai_response(user_input, document, chat_history):
    # Get the document path
    document_path = document.file.path
    file_extension = os.path.splitext(document_path)[1].lower()

    # Load document based on file type
    try:
        # Choose loader based on file extension
        if file_extension == '.pdf':
            loader = PyPDFLoader(document_path)
        elif file_extension in ['.docx', '.doc']:
            loader = Docx2txtLoader(document_path)
        else:  # Default to text loader
            loader = TextLoader(document_path)

        # Load document
        document_content = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document_content)

        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Create vector store
        vector_store = Chroma.from_documents(document_chunks, embeddings)

        # Create retriever
        retriever = vector_store.as_retriever()

        # Create LLM
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_output_tokens=800)

        # Create retriever chain
        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user",
             "Given the above conversation, generate a search query to look up information relevant to the conversation")
        ])
        retriever_chain = create_history_aware_retriever(llm, retriever, retriever_prompt)

        # Create conversational chain
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
             You are a document analysis assistant that provides information based strictly on the content of the uploaded document.
             The document content is provided here:

             {context}

             Respond to user queries by referencing only information found in the document.
             If a question cannot be answered using the document content, politely explain that the information 
             is not found in the document. Be precise and informative in your responses.
             """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])

        stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
        conversation_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)

        # Generate response
        response = conversation_chain.invoke({
            "chat_history": chat_history,
            "input": user_input
        })

        if isinstance(response, dict) and 'answer' in response:
            return response['answer']
        else:
            return "I couldn't generate a proper response based on the document. Please try a different question."

    except Exception as e:
        import traceback
        print(f"Error in document processing: {str(e)}")
        print(traceback.format_exc())
        return "I encountered an error while processing your document. Please ensure the document is readable and try again."
