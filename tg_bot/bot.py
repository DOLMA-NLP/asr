import logging
import datetime
import time
import os
import csv
import numpy as np
import subprocess
import librosa
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, BotCommand, CallbackQuery
from telegram.ext import (
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
    Application,
    ContextTypes,
    PicklePersistence,
    ConversationHandler,
)
import json
from dotenv import load_dotenv
# Define conversation states
CHOOSING_GENDER, CHOOSING_LANGUAGE, RECORDING = range(3)
RECORDING, CONFIRMING_RECORDING = range(2, 4)  
CHOOSING_LANGUAGE_COMMAND = 4 

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Load the token from environment variable
TOKEN_ID = os.getenv("TOKEN_ID")

DATASET_DIR = 'dataset'
LANG_DIR = 'languages'

SEND_TO_CHANNEL = os.getenv("SEND_TO_CHANNEL")  # Set to True to enable sending to a specific channel
CHANNEL_ID = os.getenv("CHANNEL_ID")

CHOOSING_GENDER, CHOOSING_LANGUAGE, TYPING_REPLY = range(3)

os.makedirs(DATASET_DIR, exist_ok=True)

# def load_sentences():
#     sentences = {}
#     languages = ["southern_kurdish", "laki_kurdish", "hawrami", "gilaki", "zazaki", "talysh", "mazanderani", "luri_bakhtiari"]
    
#     for lang in languages:
#         file_path = os.path.join(LANG_DIR, f"{lang}.csv")
#         metadata_path = os.path.join(DATASET_DIR, lang, "metadata.csv")
#         recorded_sentences = set()

#         # Read recorded sentences from metadata.csv
#         if os.path.exists(metadata_path):
#             with open(metadata_path, "r", encoding="utf-8") as meta_file:
#                 meta_reader = csv.reader(meta_file)
#                 next(meta_reader)  # Skip the header
#                 for row in meta_reader:
#                     recorded_sentences.add(row[1])  # Assuming the sentence is in the second column

#         try:
#             with open(file_path, "r", encoding="utf-8") as file:
#                 reader = csv.reader(file)
#                 next(reader)  # Skip the header
#                 documents = [
#                     {"english": row[0], "sentence": row[1]} 
#                     for row in reader if row[1] not in recorded_sentences
#                 ]
#                 file.seek(0)
#                 next(reader)  # Skip the header again
#                 sentences[lang] = documents 
#             logging.info(f"Loaded {len(sentences[lang])} sentences for language '{lang}'")
#         except FileNotFoundError:
#             logging.error(f"File '{file_path}' not found.")
#             sentences[lang] = []

#     return sentences

def load_sentences():
    sentences = {}
    languages = ["southern_kurdish", "laki_kurdish", "hawrami", "gilaki", "zazaki", "talysh", "mazanderani", "luri_bakhtiari"]
    
    for lang in languages:
        file_path = os.path.join(LANG_DIR, f"{lang}.csv")
        metadata_path = os.path.join(DATASET_DIR, lang, "metadata.csv")
        recorded_sentences = set()
        
        # Read recorded sentences from metadata.csv
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as meta_file:
                meta_reader = csv.reader(meta_file)
                next(meta_reader)  # Skip the header
                for row in meta_reader:
                    recorded_sentences.add(row[1])  # Assuming the sentence is in the second column
            logging.info(f"Found {len(recorded_sentences)} recorded sentences for language '{lang}'")
        
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header
                
                # Read all sentences from the language file
                all_sentences = [(row[0], row[1]) for row in reader]
                
                # Filter out already recorded sentences
                new_sentences = [
                    {"english": eng, "sentence": sent} 
                    for eng, sent in all_sentences 
                    if sent not in recorded_sentences
                ]
                
                # If new_sentences is empty (all sentences have been recorded),
                # add all sentences back to the dictionary
                if not new_sentences:
                    logging.info(f"All sentences for '{lang}' have been recorded. "
                               f"Adding all {len(all_sentences)} sentences back to the pool.")
                    sentences[lang] = [
                        {"english": eng, "sentence": sent} 
                        for eng, sent in all_sentences
                    ]
                else:
                    sentences[lang] = new_sentences
                
                logging.info(f"Language '{lang}': Total sentences: {len(all_sentences)}, "
                           f"Already recorded: {len(recorded_sentences)}, "
                           f"Final sentences in pool: {len(sentences[lang])}")
                
        except FileNotFoundError:
            logging.error(f"File '{file_path}' not found.")
            sentences[lang] = []

    return sentences

sentences_dict = load_sentences()
used_ids = []
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    context.user_data.update({
        "gender": None,
        "language": None,
        "voice_messages": [],
        "text_messages": [],
    })
    logging.info(f"User {user_id} started the bot")
    keyboard = [
        [
            InlineKeyboardButton("Male", callback_data="male"),
            InlineKeyboardButton("Female", callback_data="female"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Please select your gender", reply_markup=reply_markup)
    return CHOOSING_GENDER

async def choose_gender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    context.user_data["gender"] = query.data
    logging.info(f"User {user_id} selected gender: {query.data}")
    await prompt_language_selection(query.message, context)
    return CHOOSING_LANGUAGE

async def choose_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    context.user_data["language"] = query.data
    logging.info(f"User {user_id} selected language: {query.data}")
    if not sentences_dict.get(query.data):
        await query.message.reply_text(
            f"I'm sorry, but there are no sentences available for {query.data} at the moment. "
            "Please choose a different language."
        )
        await prompt_language_selection(query.message, context)
        return CHOOSING_LANGUAGE_COMMAND

    await send_welcome_message(query.message, context)
    await send_text_to_read(query.message, context)
    return RECORDING

async def choose_language_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    selected_language = query.data
    context.user_data["language"] = selected_language
    logging.info(f"User {user_id} changed language to: {selected_language}")

    if not sentences_dict.get(selected_language):
        await query.message.reply_text(
            f"I'm sorry, but there are no sentences available for {selected_language} at the moment. "
            "Please choose a different language."
        )
        await prompt_language_selection(query.message, context)
        return CHOOSING_LANGUAGE_COMMAND
    
    await send_text_to_read(query.message, context)
    return RECORDING

async def change_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    logging.info(f"User {user_id} requested to change language")
    await prompt_language_selection(update.message, context)
    return CHOOSING_LANGUAGE_COMMAND

async def prompt_language_selection(message, context):
    keyboard = [
        [
            InlineKeyboardButton("Hawrami", callback_data="hawrami"),
            InlineKeyboardButton("Southern Kurdish", callback_data="southern_kurdish"),
        ],
        [
            InlineKeyboardButton("Laki Kurdish", callback_data="laki_kurdish"),
            InlineKeyboardButton("Gilaki", callback_data="gilaki"),
        ],
        [
            InlineKeyboardButton("Zazaki", callback_data="zazaki"),
            InlineKeyboardButton("Talysh", callback_data="talysh"),
        ],
        [
            InlineKeyboardButton("Mazanderani", callback_data="mazanderani"),
            InlineKeyboardButton("Luri Bakhtiari", callback_data="luri_bakhtiari"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await message.reply_text("Please select your language🌐", reply_markup=reply_markup)

async def send_welcome_message(message, context):
    await message.reply_text(
        "Hi👋 Welcome!\n\n"
        "We use this bot for data gathering for research purposes📊.\n"
        "Send only one voice message per text🎤. If there's an issue with your voice message, move on to the next text➡️.\n"
        "Thank you for your collaboration🙏😊\n\n"
    )

async def send_text_to_read(message, context):
    user_id = message.chat_id
    selected_language = context.user_data["language"]
    
    if not sentences_dict.get(selected_language):
        await message.reply_text(
            f"I'm sorry, but there are no sentences available for {selected_language} at the moment. "
            "Please contact us to add more sentences"
        )
        return

    text2voice = np.random.choice(sentences_dict[selected_language])
    context.user_data["text_messages"] = text2voice
    await context.bot.send_message(
        chat_id=user_id,
        text="Please read aloud the text below (use /skip if the text is problematic or too difficult)📢:\n\n"
        + text2voice["sentence"] + "🗣️",
    )

async def skip_text(update, context):
    chat_id = update.message.chat_id
    user_language = context.user_data["language"]
    text2voice = np.random.choice(sentences_dict[user_language])
    context.user_data["text_messages"] = text2voice
    logging.info(f"User {chat_id} skipped to new text in language {user_language}")
    await context.bot.send_message(
        chat_id=chat_id,
        text="📢 Please read aloud the text below. If the text is problematic or too difficult, use /skip to get a new one:\n\n"
        + text2voice["sentence"] + "🗣️",
    )
    return RECORDING

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.message.from_user.id
    logging.info(f"User {user_id} canceled the conversation")
    await update.message.reply_text("Conversation canceled. Use /start to begin again.")
    return ConversationHandler.END

def load_voice_number():
    try:
        with open('voice_counter.json', 'r') as f:
            return json.load(f)['voice_number']
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return 100010

def save_voice_number(number):
    with open('voice_counter.json', 'w') as f:
        json.dump({'voice_number': number}, f)

VOICE_NUMBER = load_voice_number()

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    global VOICE_NUMBER
    user_id = update.message.from_user.id
    logging.info(f"Received voice message from user {user_id}")

    voice_file = await update.message.voice.get_file()
    # New file naming format using the global counter
    file_name = f"voice_{VOICE_NUMBER}.mp3"
    VOICE_NUMBER += 1  # Increment the counter
    save_voice_number(VOICE_NUMBER)  # Save the updated counter

    language_dir = os.path.join(DATASET_DIR, context.user_data["language"])
    if not os.path.exists(language_dir):
        os.makedirs(language_dir)

    file_path = os.path.join(language_dir, file_name)

    # Save voice file temporarily
    await voice_file.download_to_drive(file_path)
    context.user_data["temp_voice_message"] = file_path
    logging.info(f"Saved temporary voice message from user {user_id} to {file_path}")

    keyboard = [
        [
            InlineKeyboardButton("Accept", callback_data="accept"),
            InlineKeyboardButton("Retake", callback_data="retake"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Do you want to accept or retake the recording?",
        reply_markup=reply_markup
    )
    return CONFIRMING_RECORDING

async def confirm_recording(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if query.data == "accept":
        await accept_recording(query, context)
        await send_text_to_read(query.message, context)
        return RECORDING
    elif query.data == "retake":
        await retake_recording(query, context)
        return RECORDING

async def accept_recording(query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = query.from_user.id
    temp_file_path = context.user_data.pop("temp_voice_message")
    context.user_data["voice_messages"].append(temp_file_path)
    logging.info(f"User {user_id} accepted the recording: {temp_file_path}")

    # Get the duration of the voice message
    duration = librosa.get_duration(path=temp_file_path)



    # Save metadata to CSV
    csv_file_path = os.path.join(DATASET_DIR, context.user_data["language"], "metadata.csv")
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        if os.stat(csv_file_path).st_size == 0:
            csv_writer.writerow(["file_name", "sentence", "english", "gender", "language", "user_id", "original_full_path", "duration"])
        csv_writer.writerow([
            f"{os.path.basename(temp_file_path)}", 
            context.user_data["text_messages"]["sentence"], 
            context.user_data["text_messages"]["english"], 
            context.user_data["gender"], 
            context.user_data["language"], 
            str(user_id),
            f"{DATASET_DIR}/{context.user_data["language"]}/{os.path.basename(temp_file_path)}",
            duration
        ])

    # Send the voice message to a specific channel if enabled
    if SEND_TO_CHANNEL:
        # file_name = f"user_{user_id}_{context.user_data['language']}_{len(context.user_data['voice_messages'])}.mp3"
        caption = context.user_data["text_messages"]["sentence"]
        with open(temp_file_path, 'rb') as voice_file:
            await context.bot.send_voice(chat_id=CHANNEL_ID, voice=voice_file, caption=caption)
        logging.info(f"Sent voice message from user {user_id} to channel {CHANNEL_ID}")

    # await query.message.reply_text("Recording accepted. Here's the next text to read.")

async def retake_recording(query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = query.from_user.id
    temp_file_path = context.user_data.pop("temp_voice_message")
    os.remove(temp_file_path)
    logging.info(f"User {user_id} chose to retake the recording, deleted: {temp_file_path}")

    await query.message.reply_text(
        "Please send a new voice message for the text:\n\n"
        + context.user_data["text_messages"]["sentence"] + "🗣️",
    )

def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout) if result.returncode == 0 else 0.0


async def get_stats():
    stats = {}
    total_sentences = 0
    total_duration = 0.0

    for lang in sentences_dict.keys():
        metadata_path = os.path.join(DATASET_DIR, lang, "metadata.csv")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as meta_file:
                meta_reader = csv.reader(meta_file)
                next(meta_reader)  # Skip the header
                sentences_count = 0
                lang_duration = 0.0
                for row in meta_reader:
                    sentences_count += 1
                    lang_duration += float(row[7])  # Assuming duration is in the 8th column (index 7)
                total_sentences += sentences_count
                total_duration += lang_duration / 60.0  # Convert to minutes
                stats[lang] = {
                    "sentences": sentences_count,
                    "duration": lang_duration / 60.0  # in minutes
                }
        else:
            stats[lang] = {
                "sentences": 0,
                "duration": 0.0
            }

    return stats, total_sentences, total_duration

async def get_user_stats(user_id):
    total_sentences = 0
    total_duration = 0.0

    for lang in sentences_dict.keys():
        metadata_path = os.path.join(DATASET_DIR, lang, "metadata.csv")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as meta_file:
                meta_reader = csv.reader(meta_file)
                next(meta_reader)  # Skip the header
                for row in meta_reader:
                    if row[5] == str(user_id):  # Assuming user_id is in the 6th column (index 5)
                        total_sentences += 1
                        total_duration += float(row[7])  # Assuming duration is in the 8th column (index 7)

    return total_sentences, total_duration

# async def stats(update, context):
#     user_id = update.message.chat_id
#     stats, total_sentences, total_duration = await get_stats()

#     stats_message = f"*Total sentences labeled:* {total_sentences}\n"
#     hours = int(total_duration // 60)
#     minutes = int(total_duration % 60)
#     stats_message += f"*Total duration labeled:* {hours} hours and {minutes} minutes\n\n"
#     stats_message += "*Language-wise statistics:*\n"

#     # Use a fixed-width format for language names and values
#     max_lang_width = 20
#     for lang, data in stats.items():
#         lang = ' '.join(word.capitalize() for word in lang.split('_'))
#         stats_message += f"*{lang:<{max_lang_width}}:* {data['sentences']} sentences, {data['duration']:5.2f} minutes\n"

#     user_stats = await get_user_stats(user_id)

#     stats_message += f"\n*Your contribution:*\n"
#     stats_message += f"*Sentences:* {user_stats[0]}\n"
#     stats_message += f"*Duration:* {user_stats[1] / 60:5.2f} minutes\n"

#     # Escape special characters for MarkdownV2
    # stats_message = stats_message.replace('.', '\\.').replace('-', '\\-').replace('_', '\\_').replace('|', '\\|').replace('(', '\\(').replace(')', '\\)')

#     await context.bot.send_message(chat_id=user_id, text=stats_message, parse_mode='MarkdownV2')


async def stats(update, context):
    user_id = update.message.chat_id
    stats, total_sentences, total_duration = await get_stats()


    stats_message = f"<b>Total sentences labeled:</b> {total_sentences}\n"
    hours = int(total_duration // 60)
    minutes = int(total_duration % 60)
    stats_message += f"<b>Total duration labeled:</b> {hours:02}:{minutes:02} (HH:MM)\n\n"


    table_message = "<pre>\n"
    table_message += f"Language         | Sentence | Duration\n"
    table_message += "-----------------+----------+---------\n"
    for lang, data in stats.items():
        lang = ' '.join(word.capitalize() for word in lang.split('_'))
        hours = int(data['duration'] // 60)
        minutes = int(data['duration'] % 60)
        table_message += f"{lang:16} | {data['sentences']:8} | {hours:02}:{minutes:02}\n"
    table_message += "</pre>"


    stats_message += f"<b>Language-wise statistics:</b>\n{table_message}\n"


    user_stats = await get_user_stats(user_id)
    stats_message += f"<b>Your contribution:</b>\n"
    stats_message += f"<b>Sentences:</b> {user_stats[0]}\n"
    hours = int(user_stats[1] // 3600)
    minutes = int((user_stats[1] % 3600) // 60)
    seconds = int((user_stats[1] % 3600) % 60)
    stats_message += f"<b>Duration:</b> {hours:02}:{minutes:02}:{seconds:02} (HH:MM:SS)\n"


    await context.bot.send_message(chat_id=user_id, text=stats_message, parse_mode='HTML')

async def alarm(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the alarm message and run /skip command."""
    job = context.job
    await context.bot.send_message(job.chat_id, text="⏰ Time to label some data! Please continue with your tasks. /skip")

def remove_job_if_exists(name: str, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Remove job with given name. Returns whether job was removed."""
    current_jobs = context.job_queue.get_jobs_by_name(name)
    if not current_jobs:
        return False
    for job in current_jobs:
        job.schedule_removal()
    return True

async def set_timer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add a job to the queue."""
    chat_id = update.effective_message.chat_id
    try:
        # args[0] should contain the time in HH:MM format
        time_str = context.args[0]
        target_time = datetime.datetime.strptime(time_str, "%H:%M").time()
        now = datetime.datetime.now()
        target_datetime = datetime.datetime.combine(now.date(), target_time)

        # If the target time is already past today, set it for tomorrow
        if target_datetime < now:
            target_datetime += datetime.timedelta(days=1)

        due = (target_datetime - now).total_seconds()

        job_removed = remove_job_if_exists(str(chat_id), context)
        context.job_queue.run_once(alarm, due, chat_id=chat_id, name=str(chat_id), data=due)

        text = f"⏰ Timer successfully set for {time_str}! You will be reminded to label data."
        if job_removed:
            text += " Old one was removed."
        text += " If you want to continue labeling more data, use /skip."
        await update.effective_message.reply_text(text)

    except (IndexError, ValueError):
        await update.effective_message.reply_text("Usage: /set <HH:MM> (24-hour format)")

async def unset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove the job if the user changed their mind."""
    chat_id = update.message.chat_id
    job_removed = remove_job_if_exists(str(chat_id), context)
    text = "Timer successfully cancelled!" if job_removed else "You have no active timer."
    await update.message.reply_text(text)

async def post_init(application: Application) -> None:
    commands = [
        # BotCommand("start", "Start the bot"),
        BotCommand("skip", "Skip the current text"),
        BotCommand("change_language", "Change your language"),
        BotCommand("stats", "View dataset statistics"),
        # BotCommand("set", "Set a timer (24-hour format)"),
        # BotCommand("unset", "Unset the timer"),
        BotCommand("cancel", "Cancel the current conversation (to change gender)"),
    ]
    await application.bot.set_my_commands(commands)

def main():
    persistence = PicklePersistence(filepath="bot_data")
    application = Application.builder().token(TOKEN_ID).persistence(persistence).post_init(post_init).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHOOSING_GENDER: [CallbackQueryHandler(choose_gender)],
            CHOOSING_LANGUAGE: [CallbackQueryHandler(choose_language)],
            CHOOSING_LANGUAGE_COMMAND: [CallbackQueryHandler(choose_language_command)],
            RECORDING: [
                MessageHandler(filters.VOICE, handle_voice_message),
                CommandHandler("skip", skip_text),
                CommandHandler("change_language", change_language),
            ],
            CONFIRMING_RECORDING: [CallbackQueryHandler(confirm_recording)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        name="my_conversation",
        persistent=True,
    )

    application.add_handler(conv_handler)
    
    # These handlers are outside the conversation handler
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("set", set_timer))
    application.add_handler(CommandHandler("unset", unset))

    application.run_polling()
    logging.info("Bot stopped")

if __name__ == "__main__":
    main()