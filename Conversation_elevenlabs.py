import logging
import requests
import os
from groq import Groq
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)

import re
import time
import pygame
import io
import threading
import webrtcvad
import numpy as np
import pyaudio
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from dotenv import load_dotenv
import csv
import json
from datetime import datetime
import hashlib
from pathlib import Path
from elevenlabs import ElevenLabs, VoiceSettings

load_dotenv()  
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "deepseek-r1-distill-llama-70b"
# TTS_MODEL = "playai-tts"  
# TTS_VOICE = "Atlas-PlayAI"  
# === ElevenLabs TTS Configuration ===
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "yoZ06aMxZJJ28mfd3POQ" 

TRIGGER_WORD = "KAAZI"
TRIGGER_THRESHOLD = 70
SESSION_MODE = True

# === Conversation session management ===
CONVERSATION_TIMEOUT = 30
CONVERSATION_ACTIVE = False
LAST_INTERACTION_TIME = None
IS_PLAYING_TTS = False
last_tts_text = ""

# === Memory Management Settings ===
MEMORY_DIR = "conversation_memory"
MEMORY_FILE = "conversations.csv"
MAX_MEMORY_ENTRIES = 50  # Maximum conversation entries to keep per person
CURRENT_USER = None  # Track current user
NAME_EXTRACTION_ATTEMPTS = 0  # Track attempts to get user's name

# === Setup logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Initialize pygame mixer for audio playback ===
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# === Create memory directory ===
Path(MEMORY_DIR).mkdir(exist_ok=True)


# === Memory Management System ===
class ConversationMemory:
    def __init__(self, memory_dir=MEMORY_DIR, memory_file=MEMORY_FILE):
        self.memory_path = Path(memory_dir) / memory_file
        self.ensure_csv_exists()
        self.memory_cache = {}  # In-memory cache for faster access
        self.load_memory_cache()
    
    def ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not self.memory_path.exists():
            with open(self.memory_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['user_id', 'name', 'timestamp', 'role', 'message', 'session_id'])
    
    def generate_user_id(self, name):
        """Generate consistent user ID from name"""
        return hashlib.md5(name.lower().strip().encode()).hexdigest()[:8]
    
    def load_memory_cache(self):
        """Load recent conversations into memory cache"""
        try:
            with open(self.memory_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    user_id = row['user_id']
                    if user_id not in self.memory_cache:
                        self.memory_cache[user_id] = []
                    self.memory_cache[user_id].append(row)
        except FileNotFoundError:
            logger.info("No existing memory file found, starting fresh")
    
    def save_message(self, name, role, message, session_id=None):
        """Save a message to both CSV and cache"""
        user_id = self.generate_user_id(name)
        timestamp = datetime.now().isoformat()
        
        # Save to CSV
        with open(self.memory_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([user_id, name, timestamp, role, message, session_id or ""])
        
        # Update cache
        if user_id not in self.memory_cache:
            self.memory_cache[user_id] = []
        
        self.memory_cache[user_id].append({
            'user_id': user_id,
            'name': name,
            'timestamp': timestamp,
            'role': role,
            'message': message,
            'session_id': session_id or ""
        })
        
        # Limit memory entries per user
        if len(self.memory_cache[user_id]) > MAX_MEMORY_ENTRIES:
            self.memory_cache[user_id] = self.memory_cache[user_id][-MAX_MEMORY_ENTRIES:]
    
    def get_user_history(self, name, limit=10):
        """Get recent conversation history for a user"""
        user_id = self.generate_user_id(name)
        if user_id not in self.memory_cache:
            return []
        
        # Return recent messages
        recent_messages = self.memory_cache[user_id][-limit:]
        return [{'role': msg['role'], 'content': msg['message']} for msg in recent_messages]
    
    def user_exists(self, name):
        """Check if user exists in memory"""
        user_id = self.generate_user_id(name)
        return user_id in self.memory_cache and len(self.memory_cache[user_id]) > 0
    
    def get_user_summary(self, name):
        """Get a summary of user's conversation history"""
        user_id = self.generate_user_id(name)
        if user_id not in self.memory_cache:
            return None
        
        messages = self.memory_cache[user_id]
        if not messages:
            return None
        
        first_interaction = messages[0]['timestamp']
        last_interaction = messages[-1]['timestamp']
        total_messages = len(messages)
        
        return {
            'name': name,
            'first_interaction': first_interaction,
            'last_interaction': last_interaction,
            'total_messages': total_messages
        }
    
    def get_last_n_messages(self, name, n=5, openai_format=True):
        user_id = self.generate_user_id(name)
        if user_id not in self.memory_cache:
            return []

        sorted_msgs = sorted(self.memory_cache[user_id], key=lambda x: x['timestamp'])
        last_n = sorted_msgs[-n:]

        if openai_format:
            return [{'role': msg['role'], 'content': msg['message']} for msg in last_n]
        else:
            return last_n

    

# Initialize memory system
memory = ConversationMemory()

def get_conversation_context(user_name=None, n_messages=5):
    """Get conversation context including memory"""
    base_context = [
        {
            "role": "system",
            "content": (
                "You are KAAZI, a voice assistant that remembers previous conversations. "
                "When responding, only refer to facts and topics from saved memory. "
                "If you do not remember, say so clearly and do not make up details. "
                "Keep responses very short yet conversational and expressive."
                "IMPORTANT: Add natural emotional context for text-to-speech using these formats:"
                "Write naturally with emotional context that the AI can understand from the words themselves. "
                "Example: 'Hi there! Haha, that's so funny! I'm excited to help you today!'"
            )
        }
    ]
    
    if user_name and memory.user_exists(user_name):
        summary = memory.get_user_summary(user_name)

        if summary and "last_interaction" in summary:
            base_context.append({
                "role": "system", 
                "content": f"You are talking to {user_name}. You have spoken with them before. Your last conversation was on {summary['last_interaction'][:10]}."
            })
        else:
            base_context.append({
                "role": "system",
                "content": f"You are talking to {user_name}. You have spoken with them before."
            })

        # Add past conversation messages to context
        history = memory.get_last_n_messages(user_name, n=n_messages, openai_format=True)
        base_context.extend(history)

    return base_context


# === Enhanced Name extraction and management ===
def extract_name_from_text(text):
    """Extract name from user input using various patterns - supports international names"""
    # Enhanced patterns for better name detection
    patterns = [
        # Standard introductions
        r"(?:my name is|i'm|i am|call me|this is)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",
        r"(?:i'm|i am)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",
        r"(?:you can call me|just call me)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",
        
        # More natural patterns
        r"(?:it's|i am|i'm)\s+([a-zA-Z]+)(?:\s+speaking|$)",
        r"(?:this is|here is)\s+([a-zA-Z]+)(?:\s+speaking|$)",
        r"^([a-zA-Z]+)(?:\s+here|$)",  # Simple name at start
        
        # Question responses
        r"(?:yes|yeah|yep),?\s*(?:i'm|i am|my name is|it's)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",
        r"(?:sure|okay|ok),?\s*(?:i'm|i am|my name is|it's)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",
        
        # International name patterns
        r"(?:naam|name)\s+(?:hai|is)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",  # Hindi/Urdu
        r"(?:je suis|je m'appelle)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",  # French
        r"(?:me llamo|soy)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)",  # Spanish
    ]
    
    # Common name variations and phonetic spellings
    name_corrections = {
        'krisha': 'Krishna',
        'krishna': 'Krishna',
        'krisna': 'Krishna',
        'aman': 'Aman',
        'amaan': 'Aman',
        'kriti': 'Kriti',
        'krithi': 'Kriti',
        'arjun': 'Arjun',
        'arjun': 'Arjun',
        'priya': 'Priya',
        'priyaa': 'Priya',
        'raj': 'Raj',
        'ravi': 'Ravi',
        'dev': 'Dev',
        'devi': 'Devi',
        'maya': 'Maya',
        'maia': 'Maya',
        'sara': 'Sara',
        'sarah': 'Sarah',
        'mohamed': 'Mohamed',
        'muhammad': 'Muhammad',
        'ali': 'Ali',
        'aisha': 'Aisha',
        'fatima': 'Fatima',
        'omar': 'Omar',
        'yuki': 'Yuki',
        'hiroshi': 'Hiroshi',
        'akira': 'Akira',
        'maria': 'Maria',
        'jose': 'Jose',
        'carlos': 'Carlos',
        'ana': 'Ana',
        'diego': 'Diego',
        'sofia': 'Sofia',
        'miguel': 'Miguel',
        'chen': 'Chen',
        'wei': 'Wei',
        'ming': 'Ming',
        'xin': 'Xin',
        'jun': 'Jun',
        'yuki': 'Yuki',
        'jean': 'Jean',
        'marie': 'Marie',
        'pierre': 'Pierre',
        'claire': 'Claire',
        'antoine': 'Antoine',
    }
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw_name = match.group(1).strip()
            # Clean up the name
            name = re.sub(r'[^\w\s]', '', raw_name)  # Remove punctuation
            name = ' '.join(name.split())  # Normalize whitespace
            
            # Validate name
            if is_valid_name(name):
                # Apply corrections for common variations
                name_lower = name.lower()
                if name_lower in name_corrections:
                    return name_corrections[name_lower]
                
                # Proper case the name
                return format_name(name)
    
    # Fallback: check if the entire text might be just a name
    cleaned_text = re.sub(r'[^\w\s]', '', text).strip()
    if len(cleaned_text.split()) <= 2 and is_valid_name(cleaned_text):
        return format_name(cleaned_text)
    
    return None

def is_valid_name(name):
    """Enhanced name validation for international names"""
    if not name or len(name.strip()) < 2:
        return False
    
    # Split into parts
    parts = name.split()
    
    # Check each part
    for part in parts:
        # Must be at least 2 characters
        if len(part) < 2:
            return False
        
        # Must contain only letters (support Unicode)
        if not part.replace('-', '').replace("'", '').isalpha():
            return False
    
    # Reject common non-name words
    common_words = {
        'hello', 'hi', 'hey', 'good', 'morning', 'afternoon', 'evening', 'night',
        'yes', 'no', 'okay', 'ok', 'sure', 'thanks', 'thank', 'you', 'please',
        'what', 'when', 'where', 'how', 'why', 'who', 'can', 'will', 'would',
        'should', 'could', 'might', 'may', 'must', 'shall', 'do', 'does', 'did',
        'have', 'has', 'had', 'am', 'is', 'are', 'was', 'were', 'been', 'being',
        'help', 'assistance', 'question', 'answer', 'problem', 'issue', 'today',
        'yesterday', 'tomorrow', 'time', 'day', 'week', 'month', 'year', 'here',
        'there', 'this', 'that', 'these', 'those', 'something', 'nothing',
        'anything', 'everything', 'someone', 'anyone', 'everyone', 'nobody'
    }
    
    name_lower = name.lower()
    if name_lower in common_words:
        return False
    
    # Check if all parts are common words
    if all(part.lower() in common_words for part in parts):
        return False
    
    return True

def format_name(name):
    """Format name with proper capitalization for international names"""
    # Handle hyphenated names and apostrophes
    parts = name.split()
    formatted_parts = []
    
    for part in parts:
        if '-' in part:
            # Handle hyphenated names like "Marie-Claire"
            hyphen_parts = part.split('-')
            formatted_part = '-'.join(p.capitalize() for p in hyphen_parts)
        elif "'" in part:
            # Handle names with apostrophes like "O'Connor"
            apostrophe_parts = part.split("'")
            formatted_part = "'".join(p.capitalize() for p in apostrophe_parts)
        else:
            # Special cases for certain names
            part_lower = part.lower()
            if part_lower in ['de', 'da', 'du', 'del', 'della', 'di', 'von', 'van', 'el', 'al']:
                formatted_part = part_lower  # Keep articles lowercase
            else:
                formatted_part = part.capitalize()
        
        formatted_parts.append(formatted_part)
    
    return ' '.join(formatted_parts)

def detect_name_intent(text):
    """Detect if user is trying to tell their name even without explicit patterns"""
    # Check for name-like single words in short responses
    words = text.strip().split()
    
    if len(words) == 1:
        word = words[0]
        # Could be a name if it's capitalized or looks name-like
        if (word[0].isupper() or len(word) >= 3) and is_valid_name(word):
            return word
    
    # Check for responses to "what's your name" type questions
    name_response_patterns = [
        r"^([a-zA-Z]+)$",  # Just a name
        r"^([a-zA-Z]+)\s+([a-zA-Z]+)$",  # First and last name
        r"^it's\s+([a-zA-Z]+)$",  # "It's John"
        r"^([a-zA-Z]+)\s+is\s+my\s+name$",  # "John is my name"
    ]
    
    for pattern in name_response_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            potential_name = match.group(1)
            if is_valid_name(potential_name):
                return format_name(potential_name)
    
    return None

def should_ask_for_name():
    """Determine if we should ask for the user's name"""
    global NAME_EXTRACTION_ATTEMPTS, CURRENT_USER
    return CURRENT_USER is None and NAME_EXTRACTION_ATTEMPTS < 3

def ask_for_name():
    """Generate a natural request for the user's name"""
    global NAME_EXTRACTION_ATTEMPTS
    NAME_EXTRACTION_ATTEMPTS += 1
    
    if NAME_EXTRACTION_ATTEMPTS == 1:
        return "Hello! I'm KAAZI, What's your name? "
    elif NAME_EXTRACTION_ATTEMPTS == 2:
        return "I'd love to get to know you better. Could you tell me your name? "
    else:
        return "No worries! I'll just call you friend for now. You can tell me your name anytime. How can I help you today?"

# === Updated conversation history management ===
conversation_history = []

def reset_conversation_history():
    """Reset conversation history with current user context"""
    global conversation_history, CURRENT_USER
    conversation_history = get_conversation_context(CURRENT_USER)

def reset_user_session():
    global CURRENT_USER, NAME_EXTRACTION_ATTEMPTS, conversation_history
    CURRENT_USER = None
    NAME_EXTRACTION_ATTEMPTS = 0
    conversation_history = []  # Reset history for new user context
    logger.info("🔄 User session reset-ready for new user")

def text_to_speech(text):
    """Convert text to speech using ElevenLabs SDK and play it in real-time"""
    global IS_PLAYING_TTS, last_tts_text
    last_tts_text = text
    try:
       # process_text = process_emotion_tags(text)
        IS_PLAYING_TTS = True
        logger.info(f"🔊 Converting to speech: '{text[:50]}...'")
        
        # Generate audio using ElevenLabs SDK
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text = text,
            #text = process_text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.75,
                similarity_boost=0.75,
                style=0.4,
                use_speaker_boost=True
            )
        )
        
        # Convert generator to bytes
        audio_bytes = b''.join(audio_generator)
        
        # Play audio using pygame
        audio_data = io.BytesIO(audio_bytes)
        pygame.mixer.music.load(audio_data)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        logger.info("✅ Audio playback completed")
        time.sleep(3)
        logger.info("Post-TTS buffer complete")
        
    except Exception as e:
        logger.error(f"ElevenLabs TTS error: {e}")
    finally:
        IS_PLAYING_TTS = False

def play_audio_async(text):
    """Play audio in a separate thread to avoid blocking"""
    thread = threading.Thread(target=text_to_speech, args=(text,))
    thread.daemon = True
    thread.start()

# === Conversation session management ===
def is_conversation_active():
    """Check if conversation is still active based on timeout"""
    global CONVERSATION_ACTIVE, LAST_INTERACTION_TIME
    
    if not SESSION_MODE:
        return True
    
    if LAST_INTERACTION_TIME is None:
        return False
    
    time_since_last = time.time() - LAST_INTERACTION_TIME
    
    if time_since_last > CONVERSATION_TIMEOUT:
        CONVERSATION_ACTIVE = False
        logger.info(f"Conversation timed out after {CONVERSATION_TIMEOUT} seconds")
        return False
    
    return CONVERSATION_ACTIVE

def start_conversation():
    """Start a new conversation session"""
    global CONVERSATION_ACTIVE, LAST_INTERACTION_TIME
    CONVERSATION_ACTIVE = True
    LAST_INTERACTION_TIME = time.time()
    reset_conversation_history()  # Reset with current user context
    logger.info("Conversation session started")

def update_conversation_time():
    """Update the last interaction time"""
    global LAST_INTERACTION_TIME
    LAST_INTERACTION_TIME = time.time()

def end_conversation():
    """End the current conversation session"""
    global CONVERSATION_ACTIVE, LAST_INTERACTION_TIME, NAME_EXTRACTION_ATTEMPTS
    CONVERSATION_ACTIVE = False
    LAST_INTERACTION_TIME = None
    NAME_EXTRACTION_ATTEMPTS = 0  # Reset name extraction attempts
    reset_user_session()  # Clear current user context
    logger.info("Conversation session ended")

def check_conversation_timeout():

    global CONVERSATION_ACTIVE, LAST_INTERACTION_TIME, CURRENT_USER
    if not SESSION_MODE:
        return False
    
    if LAST_INTERACTION_TIME is None:
        return False
    
    time_since_last = time.time() - LAST_INTERACTION_TIME
    
    if time_since_last > CONVERSATION_TIMEOUT:
        CONVERSATION_ACTIVE = False
        logger.info(f"Conversation timed out after {CONVERSATION_TIMEOUT} seconds")
        # Reset user session on timeout
        reset_user_session()
        return True
    
    return False


def check_end_conversation_keywords(text):
    """Check if user wants to end the conversation"""
    end_keywords = [
        "goodbye", "bye", "see you", "talk to you later", "thanks bye", "quit", "exit", "end conversation",
        "goodbye kaazi", "bye kaazi", "thank you kaazi goodbye"
    ]
    
    text_lower = text.lower()
    for keyword in end_keywords:
        if keyword in text_lower:
            return True
    return False

# === Fuzzy trigger word detection ===
def detect_trigger_word_phonetic(text):
    """Phonetic-aware trigger detection with user transition handling"""
    if not SESSION_MODE:
        return True, text, 100
    
    # Check if conversation timed out
    if check_conversation_timeout():
        logger.info("🕐 Conversation timed out - reset for new user")
    
    if is_conversation_active():
        # During active conversation, check for trigger word (interruption)
        phonetic_variations = [
            "KAAZI", "KAZI", "KAASI", "KASI", "CAZZI", "CAZI", 
            "KAAZIE", "KAZIE", "KASSI", "CASSI", "KAAZY", "KAZY"
        ]
        
        text_upper = text.upper()
        words = [re.sub(r'[^\w]', '', word) for word in text_upper.split()]
        
        for word in words:
            for variation in phonetic_variations:
                ratio = fuzz.ratio(word, variation)
                if ratio >= TRIGGER_THRESHOLD:
                    logger.info(f"🛑 Trigger word detected during conversation - interrupting TTS")
                    
                    break
        
        return True, text, 100
    
    # Not in active conversation - need trigger word to start
    phonetic_variations = [
        "KAAZI", "KAZI", "KAASI", "KASI", "CAZZI", "CAZI", 
        "KAAZIE", "KAZIE", "KASSI", "CASSI", "KAAZY", "KAZY"
    ]
    
    text_upper = text.upper()
    words = [re.sub(r'[^\w]', '', word) for word in text_upper.split()]
    
    best_ratio = 0
    best_word_index = -1
    best_variation = ""
    
    for i, word in enumerate(words):
        for variation in phonetic_variations:
            ratio = fuzz.ratio(word, variation)
            if ratio > best_ratio:
                best_ratio = ratio
                best_word_index = i
                best_variation = variation
    
    is_triggered = best_ratio >= TRIGGER_THRESHOLD
    
    cleaned_text = text
    if is_triggered and best_word_index != -1:
        words_original = text.split()
        if best_word_index < len(words_original):
            words_original.pop(best_word_index)
            cleaned_text = ' '.join(words_original).strip()
    
    if is_triggered:
        logger.info(f"🎯 Trigger detected: '{best_variation}' ({best_ratio}% confidence)")
        if CURRENT_USER is None:
            logger.info("👤 New user session starting")
    
    return is_triggered, cleaned_text, best_ratio

# === Enhanced Groq LLM call with memory ===
def get_llm_response(prompt):
    global CURRENT_USER, conversation_history
    
    # Try to extract name from prompt if we don't have one
    if CURRENT_USER is None:
        potential_name = extract_name_from_text(prompt)
        
        # Also try intent detection for simple name responses
        if not potential_name and NAME_EXTRACTION_ATTEMPTS > 0:
            potential_name = detect_name_intent(prompt)
        
        if potential_name:
            CURRENT_USER = potential_name
            logger.info(f"👤 User identified as: {CURRENT_USER}")
            reset_conversation_history()  # Reset with new user context
            
            # Welcome back message for returning users
            if memory.user_exists(CURRENT_USER):
                summary = memory.get_user_summary(CURRENT_USER)
                welcome_back = f"Welcome back, {CURRENT_USER}! Great to see you again. How have you been?"
                return welcome_back
            else:
                # First time user
                welcome_new = f"Nice to meet you, {CURRENT_USER}! I'm KAAZI. How can I help you today?"
                return welcome_new
    
    # If we still don't have a name and should ask for it
    if should_ask_for_name():
        return ask_for_name()
    
    # Add user message to conversation history
    conversation_history.append({"role": "user", "content": prompt})
    
    # Save user message to memory
    if CURRENT_USER:
        memory.save_message(CURRENT_USER, "user", prompt)
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "messages": conversation_history,
        "model": GROQ_MODEL
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        reply = response.json()['choices'][0]['message']['content'].strip()
        filtered_reply = extract_final_response(reply)
        
        # Add assistant message to conversation history
        conversation_history.append({"role": "assistant", "content": filtered_reply})
        
        # Save assistant message to memory
        if CURRENT_USER:
            memory.save_message(CURRENT_USER, "assistant", filtered_reply)
        
        return filtered_reply
    else:
        logger.error(f"Groq API error: {response.status_code} - {response.text}")
        return "[Groq Error]"

# === AssemblyAI event handlers ===
def on_begin(client: StreamingClient, event: BeginEvent):
    print(f"\n🟢 Session started: {event.id}")
    print(f"💾 Memory system initialized - saving conversations to {MEMORY_DIR}")
    if SESSION_MODE:
        print(f"🎯 Trigger word: '{TRIGGER_WORD}' (threshold: {TRIGGER_THRESHOLD}%)")
        print(f"⏰ Conversation timeout: {CONVERSATION_TIMEOUT} seconds")
        print("💡 Say the trigger word to start a conversation")
    else:
        print("🔓 Session mode disabled - all speech will be processed")
    
    #startup_message = "Hello! I'm KAAZI, your voice assistant with memory. I can remember our conversations! I'm ready to help you!"
    #play_audio_async(startup_message)

def on_turn(client: StreamingClient, event: TurnEvent):
    global CURRENT_USER
    
    if IS_PLAYING_TTS:
        logger.info("🔊 Waiting for TTS to finish before processing input...")
        return
    
    
    if event.end_of_turn and event.turn_is_formatted:
        user_input = event.transcript.strip()
        if user_input:
            print(f"\n👤 Raw input: {user_input}")

            if should_ignore_stt(user_input, last_tts_text):
                logger.info("Ignoring STT input that matches last TTS output.")
                return
            # Check if user wants to end conversation
            if check_end_conversation_keywords(user_input):
                print("👋 Ending conversation...")
                current_user_backup = CURRENT_USER
                end_conversation()
                if current_user_backup:
                    response = f"Goodbye, {current_user_backup}! I'll remember our conversation. Say my name again if you need me."
                    memory.save_message(current_user_backup, "assistant", response)
                else:
                    response = "Goodbye! Say my name again if you need me."
                print(f"🤖 KAAZI: {response}")
                play_audio_async(response)
                return
                
            
            # Detect trigger word with fuzzy matching
            is_triggered, cleaned_text, confidence = detect_trigger_word_phonetic(user_input)
            
            if is_triggered:
                if not is_conversation_active():
                    start_conversation()
                    print(f"🔵 Conversation started (timeout: {CONVERSATION_TIMEOUT}s)")
                    if CURRENT_USER:
                        print(f"👤 Current user: {CURRENT_USER}")
                    else:
                        print("👤 No user identified yet - asking for name")
                else:
                    update_conversation_time()
                    print(f"🔵 Conversation continues...")
                    if CURRENT_USER:
                        print(f"👤 Current user: {CURRENT_USER}")
                
                if cleaned_text.strip():
                    print(f"✅ Processing: '{cleaned_text}'")
                    response = get_llm_response(cleaned_text)
                    print(f"🤖 KAAZI: {response}")
                    play_audio_async(response)
                else:
                    print("✅ Trigger detected - Hello! How can I help you?")
                    response = get_llm_response("Hello")
                    print(f"🤖 KAAZI: {response}")
                    play_audio_async(response)
                    
            else:
                if not is_conversation_active() and LAST_INTERACTION_TIME:
                    print(f"⏰ Conversation timed out - say '{TRIGGER_WORD}' to start again")
                    timeout_message = f"I'm still here! Say {TRIGGER_WORD} to start a new conversation."
                    play_audio_async(timeout_message)
                    end_conversation()
                else:
                    print(f"❌ Trigger not detected (best match: {confidence}%) - Ignoring input")
                    print(f"💡 Try saying '{TRIGGER_WORD}' followed by your question")

def on_terminated(client: StreamingClient, event: TerminationEvent):
    print(f"\n🔴 Session terminated — {event.audio_duration_seconds:.2f} seconds of audio processed.")
    print(f"💾 Conversation history saved to {MEMORY_DIR}")
    goodbye_message = "Session ended. All conversations have been saved. Goodbye!"
    play_audio_async(goodbye_message)

def on_error(client: StreamingClient, error: StreamingError):
    print(f"❌ Error occurred: {error}")
    error_message = "Sorry, I encountered an error. Please try again."
    play_audio_async(error_message)

def should_ignore_stt(user_input, last_tts_text):
    if not last_tts_text:
        return False
    ratio = fuzz.ratio(user_input.strip().lower(), last_tts_text.strip().lower())
    return ratio > 80

def extract_final_response(text):
    """Extract the final response from text that may contain thinking process"""
    thinking_patterns = [
        r'<thinking>.*?</thinking>',
        r'\*thinking\*.*?\*\/thinking\*',
        r'Let me think.*?(?=\n\n|\Z)',
        r'I need to.*?(?=\n\n|\Z)',
        r'First,.*?(?=\n\n|\Z)',
        r'Actually.*?(?=\n\n|\Z)',
    ]
    
    cleaned_text = text
    for pattern in thinking_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    final_answer_patterns = [
        r'(?:Final answer|Final response|Answer):\s*(.*?)(?:\n\n|\Z)',
        r'(?:In conclusion|To summarize|So):\s*(.*?)(?:\n\n|\Z)',
        r'(?:Therefore|Thus|Hence):\s*(.*?)(?:\n\n|\Z)',
    ]
    
    for pattern in final_answer_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
    if paragraphs:
        final_paragraphs = []
        for p in paragraphs:
            if not re.match(r'^(Let me|I need to|First|Actually|Hmm|Well)', p, re.IGNORECASE):
                final_paragraphs.append(p)
        
        if final_paragraphs:
            return final_paragraphs[-1]
    
    cleaned_text = cleaned_text.strip()
    if cleaned_text:
        return cleaned_text
    
    return text.strip()

# === Main function ===
def main():
    print("🎙️ Initializing KAAZI Voice Assistant with Memory & TTS...")
    print("📦 Required packages: pygame, requests, groq, assemblyai, fuzzywuzzy")
    print(f"💾 Memory system: {MEMORY_DIR}")
    
    try:
        import pygame
        print("✅ pygame loaded successfully")
    except ImportError:
        print("❌ pygame not found. Install with: pip install pygame")
        return
    
    aai.settings.api_key = ASSEMBLYAI_API_KEY

    client = StreamingClient(
        StreamingClientOptions(
            api_key=ASSEMBLYAI_API_KEY,
            api_host="streaming.assemblyai.com",
        )
    )

    client.on(StreamingEvents.Begin, on_begin)
    client.on(StreamingEvents.Turn, on_turn)
    client.on(StreamingEvents.Termination, on_terminated)
    client.on(StreamingEvents.Error, on_error)

    client.connect(
        StreamingParameters(
            sample_rate=16000,
            format_turns=True,
            end_of_turn_confidence_threshold=0.9,
            min_end_of_turn_silence_when_confident=160,
            max_turn_silence=240,
        )
    )

    try:
        mic_stream = aai.extras.MicrophoneStream(sample_rate=16000)
        print("🎙️ Speak into your microphone... (Ctrl+C to stop)")
        if SESSION_MODE:
            print(f"🎯 Say '{TRIGGER_WORD}' to start a conversation!")
            print(f"💡 After starting, you can continue talking for {CONVERSATION_TIMEOUT} seconds without saying '{TRIGGER_WORD}' again")
            print("💡 Say 'goodbye' or 'bye' to end the conversation")
        print("🔊 TTS enabled - KAAZI will speak back to you!")
        print("🧠 Memory enabled - KAAZI will remember your conversations!")
        client.stream(mic_stream)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        farewell_message = "Goodbye! All conversations have been saved. It was nice talking with you."
        text_to_speech(farewell_message)
    finally:
        client.disconnect(terminate=True)
        pygame.mixer.quit()

if __name__ == "__main__":
    main()