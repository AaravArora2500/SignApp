from flask import Flask, render_template, send_file, jsonify
from flask_sock import Sock
import os
import logging
from pathlib import Path
import base64
import io
import numpy as np
import cv2
from PIL import Image
import pickle
import mediapipe as mp
from collections import deque, Counter
import pyttsx3
from threading import Lock, Thread, Event
import queue
import time
import requests
import re
import json

app = Flask(__name__)
sock = Sock(app)

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
DOWNLOADS_FOLDER = Path.cwd() / 'downloads'
logger.info(DOWNLOADS_FOLDER)
DOWNLOADS_FOLDER.mkdir(exist_ok=True)

INSTALLER_NAME = 'Sign Langauge Gestures Setup 1.0.0.exe'
INSTALLER_PATH = DOWNLOADS_FOLDER / INSTALLER_NAME
logger.info(INSTALLER_PATH)

print(f"Looking for installer at: {INSTALLER_PATH.absolute()}")
print(f"Installer exists: {INSTALLER_PATH.exists()}")
if DOWNLOADS_FOLDER.exists():
    print(f"Files in downloads folder: {list(DOWNLOADS_FOLDER.iterdir())}")

# ============================================
# ML MODEL SETUP
# ============================================

# Map labels (A–Z + ENTER + SPACE + THUMBS_DOWN)
LABELS_DICT = {i: chr(65 + i) for i in range(26)}
LABELS_DICT[26] = "ENTER"
LABELS_DICT[27] = "SPACE"
LABELS_DICT[28] = "THUMBS_DOWN"  # NEW: Thumbs down gesture

MODEL_PATH = 'model.p'
model = None
mp_hands = None
hands = None
tts_engine = None
tts_lock = Lock()

# Enhanced accuracy settings
BUFFER_SIZE = 3
CONFIDENCE_THRESHOLD = 0.60
STABLE_FRAMES_REQUIRED = 12
STABLE_FRAMES_ENTER = 7

# Hand stability tracking
hand_position_buffer = deque(maxlen=10)

def load_ml_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            return False
        
        logger.info("Loading RandomForest ASL model...")
        model_dict = pickle.load(open(MODEL_PATH, 'rb'))
        model = model_dict['model']
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"✅ Recognizing {len(LABELS_DICT)} gestures: {', '.join(LABELS_DICT.values())}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def init_mediapipe():
    global mp_hands, hands
    try:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.70
        )
        logger.info("✅ MediaPipe Hands initialized (Enhanced accuracy)")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MediaPipe: {e}")
        return False

def init_tts():
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 100)
        logger.info("✅ Text-to-speech initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize TTS: {e}")
        return False

def process_landmarks(hand_landmarks):
    try:
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        
        wrist_x, wrist_y = x_coords[0], y_coords[0]
        x_coords = [x - wrist_x for x in x_coords]
        y_coords = [y - wrist_y for y in y_coords]
        
        data_aux = []
        for x, y in zip(x_coords, y_coords):
            data_aux.extend([x, y])
        
        feature_count = len(data_aux)
        if feature_count == 42:
            logger.debug(f" ✓ Extracted {feature_count} features")
            return data_aux
        else:
            logger.warning(f" ✗ Feature mismatch: expected 42, got {feature_count}")
            return None
    except Exception as e:
        logger.error(f"Error processing landmarks: {e}")
        return None

def is_hand_stable(hand_landmarks, threshold=0.02):
    """Check if hand is stable (not moving)"""
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    
    hand_position_buffer.append((wrist_x, wrist_y))
    
    if len(hand_position_buffer) < 5:
        return False
    
    positions = np.array(list(hand_position_buffer))
    variance = np.var(positions, axis=0).sum()
    is_stable = variance < threshold
    
    logger.debug(f" 🎯 Hand: {'STABLE' if is_stable else 'MOVING'} (var: {variance:.4f})")
    return is_stable

def predict_sign(image_data, pred_buffer, frame_count, local_hand_buffer):
    try:
        if model is None or hands is None:
            logger.error(" ✗ Model/MediaPipe not initialized")
            return None, 0.0, False, False
        
        # Handle base64 or raw bytes
        if isinstance(image_data, str):
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
        else:
            img_bytes = image_data
        
        img = Image.open(io.BytesIO(img_bytes))
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Apply Gaussian blur to reduce noise
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
        
        results = hands.process(img_array)
        
        if results.multi_hand_landmarks:
            logger.debug(f" 👋 Frame {frame_count}: Hand detected")
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_stable = is_hand_stable(hand_landmarks)
            
            features = process_landmarks(hand_landmarks)
            if features:
                prediction = model.predict([np.asarray(features)])[0]
                pred_buffer.append(prediction)
                
                most_common, count = Counter(pred_buffer).most_common(1)[0]
                gesture = LABELS_DICT[int(most_common)]
                confidence = count / len(pred_buffer)
                
                logger.debug(f" 🔍 Pred: '{gesture}' ({confidence:.0%}, {len(pred_buffer)}/{BUFFER_SIZE}, {'STABLE' if hand_stable else 'MOVING'})")
                return gesture, confidence, True, hand_stable
            else:
                logger.warning(f" ⚠️ Frame {frame_count}: Hand detected but feature extraction failed")
        else:
            if frame_count % 30 == 0:
                logger.debug(f" ✗ Frame {frame_count}: No hand")
        
        return None, 0.0, False, False
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, 0.0, False, False

def text_to_speech(text):
    try:
        logger.info(f" 🔊 TTS: Generating audio for '{text}'")
        with tts_lock:
            if tts_engine is None:
                logger.error(" ✗ TTS engine not initialized")
                return None
            
            temp_file = "temp_audio.wav"
            tts_engine.save_to_file(text, temp_file)
            tts_engine.runAndWait()
            
            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    audio_bytes = f.read()
                os.remove(temp_file)
                logger.info(f" ✓ TTS: Audio ready ({len(audio_bytes)} bytes)")
                return audio_bytes
            
            logger.error(" ✗ TTS: Failed to create audio")
            return None
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

# ============================================
# LLM INTEGRATION
# ============================================

def get_llm_completion(current_text, conversation_history=[], sentence=""):
    """Call Ollama API for word completion with conversation context"""
    stripped_text = current_text.strip()
    
    # Build context from conversation history
    context = ""
    if conversation_history:
        recent_messages = conversation_history[-5:]  # Use last 5 messages for context
        context = "\n".join([f"- {msg}" for msg in recent_messages])
        context = f"\nPrevious conversation context:\n{context}\n"
    
    # Add current sentence to context
    if sentence.strip():
        context += f"\nCurrent incomplete sentence: \"{sentence.strip()}\"\n"
    
    # Check if we have enough context to make a prediction
    if len(stripped_text) < 2 and not sentence.strip() and not conversation_history:
        logger.debug(f" ⊘ LLM: Insufficient context")
        return ""
    
    # Build prompt based on what information we have
    if stripped_text:
        # We have a partial word
        prompt = f"""You are an autocomplete assistant for sign language text input. Given a partial word and context, predict the most likely full word it is intended to be. Use the conversation history and current sentence to understand context and make a better prediction.

Do not add any extra words, explanations, or punctuation. Return ONLY the single, most likely completed word.
{context}
Example 1:
Current sentence: "I WANT TO GO"
Input: 'HOM'
Output: 'HOME'

Example 2:
Input: 'HELPO'
Output: 'HELLO'

Example 3:
Current sentence: "MY NAME IS"
Input: 'J'
Output: 'JOHN'

Example 4:
Input: 'HOP'
Output:'HOW'

Current partial word: "{stripped_text}"
Completed word:"""
    else:
        # No current word, predict next word based on sentence
        prompt = f"""You are a predictive text assistant for sign language input. Given a conversation context and an incomplete sentence, predict the most likely NEXT word. Use the conversation history to understand context.

Do not add any extra words, explanations, or punctuation. Return ONLY the single, most likely next word use the examples above and below as.
{context}
Example 1:
Current sentence: "I WANT TO"
Output: 'GO'

Example 2:
Current sentence: "MY NAME"
Output: 'IS'

Example 3:
Previous: "HOW ARE YOU"
Current sentence: "I AM"
Output: 'GOOD'

Predict the next word:"""
    
    try:
        logger.info(f" 🤖 LLM: Requesting {'completion' if stripped_text else 'next word'} (word='{stripped_text}', sentence='{sentence.strip()}')")
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'phi3:mini',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'num_predict': 20,
                    'temperature': 0.2,
                }
            },
            timeout=10.0
        )
        
        response.raise_for_status()
        data = response.json()
        completed_word = data.get('response', '').strip().upper()
        
        logger.info(f" 📨 LLM raw response: '{completed_word}'")
        
        completed_word = re.split(r'[\s\.\?\!\n]', completed_word)[0].strip()
        
        if completed_word:
            logger.info(f" ✓ LLM: '{completed_word}'")
        else:
            logger.info(f" ⊘ LLM: No completion")
        
        return completed_word
    
    except requests.exceptions.Timeout:
        logger.warning(f" ⏱️ LLM timeout (10s)")
        return ""
    except requests.exceptions.ConnectionError:
        logger.warning(f" ✗ LLM: Cannot connect to Ollama")
        logger.warning(f" 💡 TIP: Start with 'ollama serve'")
        return ""
    except Exception as e:
        logger.warning(f" ✗ LLM error: {e}")
        return ""

# ============================================
# WEB ROUTES
# ============================================

@app.route('/')
def home():
    installer_exists = INSTALLER_PATH.exists()
    installer_size = None
    if installer_exists:
        size_bytes = INSTALLER_PATH.stat().st_size
        installer_size = f"{size_bytes / (1024 * 1024):.2f} MB"
    
    return render_template('index.html',
                         installer_exists=installer_exists,
                         installer_size=installer_size,
                         installer_name=INSTALLER_NAME)

@app.route('/overlay')
def overlay():
    """Serve the overlay window"""
    return render_template('overlay.html')

@app.route('/download')
def download():
    if not INSTALLER_PATH.exists():
        return jsonify({'error': 'Installer not found'}), 404
    
    logger.info(f"Serving installer: {INSTALLER_NAME}")
    return send_file(
        INSTALLER_PATH,
        as_attachment=True,
        download_name=INSTALLER_NAME,
        mimetype='application/octet-stream'
    )

@app.route('/api/installer-info')
def installer_info():
    if not INSTALLER_PATH.exists():
        return jsonify({'available': False, 'message': 'Installer not available'}), 404
    
    size_bytes = INSTALLER_PATH.stat().st_size
    return jsonify({
        'available': True,
        'filename': INSTALLER_NAME,
        'size_bytes': size_bytes,
        'size_mb': round(size_bytes / (1024 * 1024), 2),
        'version': '1.0.0'
    })

@app.route('/api/model-status')
def model_status():
    return jsonify({
        'model_loaded': model is not None,
        'mediapipe_loaded': hands is not None,
        'tts_available': tts_engine is not None,
        'model_type': 'RandomForest + MediaPipe + LLM',
        'gestures': list(LABELS_DICT.values()) if model else []
    })

# ============================================
# WEBSOCKET WITH STATE UPDATES
# ============================================

def frame_receiver(ws, frame_queue, control_queue, stop_event):
    """Receives frames AND control messages"""
    logger.info(" [Receiver] Thread started")
    frame_count = 0
    
    try:
        while not stop_event.is_set():
            data = ws.receive()
            if data is None:
                break
            
            if isinstance(data, str):
                try:
                    control_msg = json.loads(data)
                    if 'type' in control_msg:
                        control_queue.put_nowait(control_msg)
                        logger.info(f" [Receiver] ⌨️ Control: {control_msg['type']}")
                        continue
                except json.JSONDecodeError:
                    pass
                
                try:
                    frame_queue.put_nowait(data)
                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.debug(f" [Receiver] 📹 {frame_count} frames")
                except queue.Full:
                    logger.debug(f" [Receiver] ⚠️ Queue full, dropping frame")
            
            elif isinstance(data, bytes):
                try:
                    frame_queue.put_nowait(data)
                    frame_count += 1
                except queue.Full:
                    pass
    
    except Exception as e:
        logger.error(f" [Receiver] Error: {e}")
    finally:
        stop_event.set()
        logger.info(f" [Receiver] Stopping ({frame_count} frames)")

def send_state_update(ws, current_word, sentence):
    """Send current state to frontend"""
    try:
        state = {
            'type': 'STATE_UPDATE',
            'word': current_word,
            'sentence': sentence.strip(),
            'timestamp': time.time()
        }
        ws.send(json.dumps(state))
        logger.debug(f" [State] Sent: word='{current_word}', sentence='{sentence.strip()}'")
    except Exception as e:
        logger.error(f" [State] Error sending: {e}")

def frame_processor(ws, frame_queue, control_queue, stop_event):
    """Processes frames and handles controls"""
    logger.info(" [Processor] Thread started")
    logger.info(" [Processor] 📝 State: word='', sentence=''")
    
    current_word = ""
    sentence = ""
    last_gesture = ""
    same_gesture_count = 0
    
    # NEW: Conversation history storage
    conversation_history = []
    
    # Double letter (2.5s hold)
    double_letter_frames_required = 15
    letter_hold_count = 0
    letter_already_doubled = False
    no_hand_frames = 0
    
    # Double letter prevention
    last_confirmed_gesture = None
    last_confirmed_time = 0
    gesture_cooldown = 2.5
    
    # Pause state
    is_paused = False
    
    # Statistics
    total_predictions = 0
    successful_predictions = 0
    
    pred_buffer = deque(maxlen=BUFFER_SIZE)
    local_hand_buffer = deque(maxlen=10)
    frame_count = 0
    hand_detected_frames = 0
    
    # Send initial state
    send_state_update(ws, current_word, sentence)
    
    try:
        while not stop_event.is_set():
            # Handle control messages
            try:
                control = control_queue.get_nowait()
                
                # PAUSE/RESUME
                if control['type'] == 'PAUSE':
                    is_paused = True
                    logger.info(" [Processor] " + "="*50)
                    logger.info(" [Processor] ⏸️ PAUSED - Frame processing stopped")
                    logger.info(" [Processor] " + "="*50)
                
                elif control['type'] == 'RESUME':
                    is_paused = False
                    logger.info(" [Processor] " + "="*50)
                    logger.info(" [Processor] ▶️ RESUMED - Frame processing active")
                    logger.info(" [Processor] " + "="*50)
                
                # BACKSPACE
                elif control['type'] == 'BACKSPACE':
                    logger.info(" [Processor] " + "="*50)
                    logger.info(" [Processor] ⌨️ BACKSPACE pressed")
                    logger.info(f" [Processor] 📝 Current word: '{current_word}'")
                    
                    if current_word:
                        current_word = current_word[:-1]
                        logger.info(f" [Processor] ✂️ Deleted last character")
                        logger.info(f" [Processor] 📝 New word: '{current_word}'")
                        send_state_update(ws, current_word, sentence)
                        logger.info(f" [Processor] 📤 State updated")
                        
                        # Reset gesture tracking
                        last_gesture = ""
                        same_gesture_count = 0
                        letter_hold_count = 0
                        letter_already_doubled = False
                    else:
                        logger.info(f" [Processor] ⊘ No character to delete")
                    
                    logger.info(" [Processor] " + "="*50)
                
                elif control['type'] == 'AUTOCORRECT':
                    logger.info(" [Processor] " + "="*50)
                    logger.info(" [Processor] 🔤 AUTOCORRECT pressed")
                    logger.info(f" [Processor] 📝 Current word: '{current_word}'")
                    logger.info(f" [Processor] 📄 Current sentence: '{sentence.strip()}'")
                    logger.info(f" [Processor] 📚 Context: {len(conversation_history)} previous messages")
                    
                    completed_word = get_llm_completion(current_word, conversation_history, sentence)
                    
                    if completed_word:
                        if current_word:
                            logger.info(f" [Processor] ✨ '{current_word}' → '{completed_word}'")
                        else:
                            logger.info(f" [Processor] ✨ Predicted next word: '{completed_word}'")
                        
                        current_word = completed_word
                        send_state_update(ws, current_word, sentence)
                        logger.info(f" [Processor] 📤 State updated")
                    else:
                        logger.info(f" [Processor] ⊘ No completion available")
                    
                    if current_word:
                        audio = text_to_speech(current_word)
                        if audio:
                            ws.send(audio)
                    
                    logger.info(" [Processor] " + "="*50)
                
                elif control['type'] == 'SPACE':
                    logger.info(" [Processor] " + "="*50)
                    logger.info(" [Processor] ⌨️ SPACE pressed")
                    logger.info(f" [Processor] 📝 Current word: '{current_word}'")
                    
                    if current_word:
                        sentence += current_word + " "
                        logger.info(f" [Processor] ✓ Added '{current_word}'")
                        logger.info(f" [Processor] 📄 Sentence: '{sentence}'")
                        current_word = ""
                        send_state_update(ws, current_word, sentence)
                        logger.info(f" [Processor] 📤 State updated")
                    else:
                        logger.info(f" [Processor] ⊘ No word to add")
                    
                    # Reset gesture tracking state
                    last_gesture = ""
                    same_gesture_count = 0
                    letter_hold_count = 0
                    letter_already_doubled = False
                    no_hand_frames = 0
                    
                    logger.info(" [Processor] " + "="*50)
                
                elif control['type'] == 'SUBMIT_SENTENCE':
                    logger.info(" [Processor] " + "="*50)
                    logger.info(" [Processor] ⌨️ ENTER pressed - Submitting sentence")
                    
                    if current_word:
                        sentence += current_word + " "
                        logger.info(f" [Processor] ✓ Added final word '{current_word}'")
                    
                    final_sentence = sentence.strip()
                    if final_sentence:
                        logger.info(f" [Processor] 📢 SUBMITTING: '{final_sentence}'")
                        
                        # Store in conversation history
                        conversation_history.append(final_sentence)
                        logger.info(f" [Processor] 💾 Saved to history (Total: {len(conversation_history)} messages)")
                        
                        audio = text_to_speech(final_sentence)
                        if audio:
                            ws.send(audio)
                            logger.info(f" [Processor] 🔊 Sentence audio sent")
                        
                        sentence = ""
                        current_word = ""
                        last_gesture = ""
                        same_gesture_count = 0
                        letter_hold_count = 0
                        letter_already_doubled = False
                        no_hand_frames = 0
                        
                        send_state_update(ws, current_word, sentence)
                        logger.info(f" [Processor] 🔄 State reset")
                    else:
                        logger.info(" [Processor] ⊘ No sentence to submit")
                    
                    logger.info(" [Processor] " + "="*50)
                
                elif control['type'] == 'GET_STATE':
                    send_state_update(ws, current_word, sentence)
                
                elif control['type'] == 'RESET':
                    logger.info(" [Processor] " + "="*50)
                    logger.info(" [Processor] 🔄 RESET pressed")
                    current_word = ""
                    sentence = ""
                    last_gesture = ""
                    same_gesture_count = 0
                    letter_hold_count = 0
                    letter_already_doubled = False
                    no_hand_frames = 0
                    send_state_update(ws, current_word, sentence)
                    logger.info(f" [Processor] 📤 State reset")
                    logger.info(" [Processor] " + "="*50)
            
            except queue.Empty:
                pass
            
            # Skip frame processing if paused
            if is_paused:
                time.sleep(0.1)
                continue
            
            # Process frames (only if not paused)
            try:
                frame_data = frame_queue.get(block=True, timeout=0.1)
                frame_count += 1
                
                gesture, confidence, hand_detected, hand_stable = predict_sign(
                    frame_data, pred_buffer, frame_count, local_hand_buffer
                )
                
                if hand_detected:
                    hand_detected_frames += 1
                    no_hand_frames = 0
                    stability = "🎯 STABLE" if hand_stable else "📍 MOVING"
                    logger.info(f" [Processor] {stability} | '{gesture}' ({confidence:.0%}) | Word: '{current_word}' | Sentence: '{sentence.strip()}'")
                else:
                    no_hand_frames += 1
                    if frame_count % 30 == 0:
                        logger.debug(f" [Processor] ✗ No hand ({no_hand_frames}f) | Word: '{current_word}'")
                
                # Stats every 100 frames
                if frame_count % 100 == 0:
                    rate = (hand_detected_frames / 100) * 100
                    success_rate = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0
                    logger.info(f" [Processor] 📈 Hand: {rate:.0f}% | Success: {success_rate:.0f}% | Frames: {frame_count}")
                    hand_detected_frames = 0
                
                # Gesture confirmation - requires stability
                if gesture and confidence >= CONFIDENCE_THRESHOLD and hand_stable:
                    total_predictions += 1
                    
                    if gesture == last_gesture:
                        same_gesture_count += 1
                        logger.debug(f" [Processor] 🔁 '{gesture}' x{same_gesture_count}/{STABLE_FRAMES_REQUIRED}")
                        
                        # Double letter check (only for regular letters, not SPACE/ENTER/THUMBS_DOWN)
                        if gesture not in ["SPACE", "ENTER", "THUMBS_DOWN"] and current_word and gesture == current_word[-1]:
                            letter_hold_count += 1
                            if letter_hold_count >= double_letter_frames_required and not letter_already_doubled:
                                current_time = time.time()
                                if (last_confirmed_gesture != gesture or 
                                    current_time - last_confirmed_time >= gesture_cooldown):
                                    
                                    current_word += gesture
                                    logger.info(" [Processor] " + "="*50)
                                    logger.info(f" [Processor] 🔁 DOUBLE: '{gesture}' held 2.5s")
                                    logger.info(f" [Processor] 📝 Word: '{current_word}'")
                                    logger.info(" [Processor] " + "="*50)
                                    
                                    send_state_update(ws, current_word, sentence)
                                    letter_already_doubled = True
                                    letter_hold_count = 0
                                    last_confirmed_gesture = gesture
                                    last_confirmed_time = current_time
                    else:
                        if last_gesture:
                            logger.debug(f" [Processor] ⏭️ '{last_gesture}' → '{gesture}'")
                        last_gesture = gesture
                        same_gesture_count = 1
                        letter_hold_count = 0
                        letter_already_doubled = False
                    
   
                    # Confirm new gesture
                    if same_gesture_count >= STABLE_FRAMES_ENTER:
                        current_time = time.time()
                        
                        # Prevent duplicate confirmations
                        if (last_confirmed_gesture != gesture or 
                            current_time - last_confirmed_time >= gesture_cooldown):
                            
                            # NEW: Handle THUMBS_DOWN gesture
                            if gesture == "THUMBS_DOWN":
                                logger.info(" [Processor] " + "="*50)
                                logger.info(" [Processor] ✅ THUMBS DOWN DETECTED")
                                logger.info(f" [Processor] 📝 Current word: '{current_word}'")
                                
                                if current_word:
                                    current_word = current_word[:-1]
                                    logger.info(f" [Processor] 👎 Deleted last character")
                                    logger.info(f" [Processor] 📝 New word: '{current_word}'")
                                    send_state_update(ws, current_word, sentence)
                                    successful_predictions += 1
                                else:
                                    logger.info(f" [Processor] ⊘ No character to delete")
                                
                                logger.info(" [Processor] " + "="*50)
                            
                            # Handle ENTER gesture
                            elif gesture == "ENTER":
                                logger.info(" [Processor] " + "="*50)
                                logger.info(" [Processor] ✅ ENTER SIGN DETECTED")
                                
                                # Add current word if exists
                                if current_word:
                                    sentence += current_word + " "
                                    logger.info(f" [Processor] ✓ Added final word '{current_word}'")
                                
                                final_sentence = sentence.strip()
                                if final_sentence:
                                    logger.info(f" [Processor] 📢 Speaking: '{final_sentence}'")
                                    
                                    # Store in conversation history
                                    conversation_history.append(final_sentence)
                                    logger.info(f" [Processor] 💾 Saved to history (Total: {len(conversation_history)} messages)")
                                    
                                    audio = text_to_speech(final_sentence)
                                    if audio:
                                        ws.send(audio)
                                        logger.info(f" [Processor] 🔊 Sentence audio sent")
                                    
                                    sentence = ""
                                    current_word = ""
                                    send_state_update(ws, current_word, sentence)
                                else:
                                    logger.info(" [Processor] ⊘ No sentence to speak")
                                
                                logger.info(" [Processor] " + "="*50)
                                successful_predictions += 1
                            
                            # Handle SPACE gesture
                            elif gesture == "SPACE":
                                logger.info(" [Processor] " + "="*50)
                                logger.info(" [Processor] ✅ SPACE SIGN DETECTED")
                                
                                if current_word:
                                    sentence += current_word + " "
                                    logger.info(f" [Processor] ✓ Added '{current_word}' to sentence")
                                    logger.info(f" [Processor] 📄 Sentence: '{sentence}'")
                                    current_word = ""
                                    send_state_update(ws, current_word, sentence)
                                else:
                                    logger.info(f" [Processor] ⊘ No word to add")
                                
                                logger.info(" [Processor] " + "="*50)
                                successful_predictions += 1
                            
                            # Handle regular letters (A-Z)
                            elif not current_word or gesture != current_word[-1]:
                                current_word += gesture
                                successful_predictions += 1
                                
                                logger.info(" [Processor] " + "="*50)
                                logger.info(f" [Processor] ✅ CONFIRMED: '{gesture}'")
                                logger.info(f" [Processor] 📝 Word: '{current_word}'")
                                logger.info(f" [Processor] 📄 Sentence: '{sentence.strip()}'")
                                logger.info(f" [Processor] 📊 Success: {(successful_predictions/total_predictions)*100:.0f}%")
                                logger.info(" [Processor] " + "="*50)
                                
                                send_state_update(ws, current_word, sentence)
                            
                            last_confirmed_gesture = gesture
                            last_confirmed_time = current_time
                            same_gesture_count = 0
                            letter_hold_count = 0
                            letter_already_doubled = False
                
                elif gesture and confidence >= CONFIDENCE_THRESHOLD and not hand_stable:
                    logger.debug(f" [Processor] ⚠️ '{gesture}' detected but not stable")
                
                else:
                    if not hand_detected:
                        if last_gesture:
                            logger.debug(f" [Processor] 🔄 Reset tracking")
                            last_gesture = ""
                            same_gesture_count = 0
                            letter_hold_count = 0
                            letter_already_doubled = False
                            local_hand_buffer.clear()
            
            except queue.Empty:
                if stop_event.is_set():
                    break
            except Exception as e:
                logger.error(f" [Processor] Error: {e}", exc_info=True)
    
    finally:
        success_rate = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0
        logger.info(f" [Processor] Stopping")
        logger.info(f" [Processor] 📊 Stats: {successful_predictions}/{total_predictions} ({success_rate:.0f}%)")
        logger.info(f" [Processor] 📄 Final: '{sentence.strip()}'")
        logger.info(f" [Processor] 📝 Word: '{current_word}'")
        logger.info(f" [Processor] 💾 Conversation history: {len(conversation_history)} messages")
        
        # Log conversation history for debugging
        if conversation_history:
            logger.info(" [Processor] 📚 History:")
            for i, msg in enumerate(conversation_history[-10:], 1):  # Show last 10
                logger.info(f"   [Processor]   {i}. {msg}")

@sock.route('/ws/ml')
def ml_websocket(ws):
    logger.info("="*60)
    logger.info("🔌 New WebSocket connection")
    logger.info("="*60)
    
    if model is None or hands is None:
        logger.error("✗ Model not loaded!")
        ws.send(json.dumps({'error': 'Model not loaded'}))
        return
    
    logger.info("✓ Model ready")
    
    frame_queue = queue.LifoQueue(maxsize=5)
    control_queue = queue.Queue()
    stop_event = Event()
    
    receiver = Thread(target=frame_receiver, args=(ws, frame_queue, control_queue, stop_event))
    processor = Thread(target=frame_processor, args=(ws, frame_queue, control_queue, stop_event))
    
    receiver.start()
    processor.start()
    
    logger.info("✓ Worker threads started")
    
    try:
        receiver.join()
    except Exception as e:
        logger.error(f"Error: {e}")
    
    stop_event.set()
    processor.join(timeout=5)
    
    logger.info("="*60)
    logger.info("🔌 Connection closed")
    logger.info("="*60)

# ============================================
# ERROR HANDLERS
# ============================================

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ASL to Audio Server (Enhanced + Overlay + Thumbs Down)")
    print("="*60)
    
    if INSTALLER_PATH.exists():
        logger.info(f"✅ Installer found")
    else:
        logger.warning(f"⚠️ Installer not found")
    
    model_loaded = load_ml_model()
    mediapipe_loaded = init_mediapipe()
    tts_loaded = init_tts()
    
    print("="*60)
    print("💡 Features:")
    print("   • Model detects: A-Z + SPACE + ENTER + THUMBS_DOWN gestures")
    print("   • THUMBS DOWN sign → Delete last character")
    print("   • SPACE sign → Add word to sentence")
    print("   • ENTER sign → Speak entire sentence")
    print("   • P key → Pause/Resume video processing")
    print("   • Backspace → Delete last character")
    print("   • A key → Autocorrect with LLM")
    print("   • Manual SPACE/ENTER keys still work")
    print("   • Real-time state updates to frontend")
    print("   • Conversation history tracking")
    print("="*60)
    print("📥 Main UI: http://localhost:8080")
    print("🎯 Overlay: http://localhost:8080/overlay")
    print("🔌 WebSocket: ws://localhost:8080/ws/ml")
    print("="*60)
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False)