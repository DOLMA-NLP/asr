import asyncio
import pandas as pd
import os
import argparse
from telegram import Bot
from telegram.error import Forbidden, RetryAfter
from plot import plot_progress_over_time
from dotenv import load_dotenv

load_dotenv()

# Your bot token here
TOKEN_ID = os.getenv("TOKEN_ID")
DATASET_DIR = 'dataset'
_TEST_ID = "1234567890"

# Initialize the bot
bot = Bot(token=TOKEN_ID)

def format_duration(seconds):
    """
    Convert seconds to hours and round to 1 decimal place
    """
    return round(seconds / 3600, 1)

async def safe_send_message(bot, chat_id, text):
    """
    Safely send a message with retry logic for rate limits
    """
    while True:
        try:
            await bot.send_message(chat_id=str(chat_id), text=text)
            return
        except RetryAfter as e:
            print(f"Rate limit hit. Waiting {e.retry_after} seconds...")
            await asyncio.sleep(e.retry_after)
        except Exception as e:
            print(f"Error sending message to {chat_id}: {str(e)}")
            return

async def safe_send_photo(bot, chat_id, photo_path):
    """
    Safely send a photo with retry logic for rate limits
    """
    while True:
        try:
            with open(photo_path, 'rb') as photo:
                await bot.send_photo(chat_id=str(chat_id), photo=photo)
            return
        except RetryAfter as e:
            print(f"Rate limit hit. Waiting {e.retry_after} seconds...")
            await asyncio.sleep(e.retry_after)
        except Exception as e:
            print(f"Error sending photo to {chat_id}: {str(e)}")
            return

async def send_targeted_message(language, message):
    """
    Send a specific message to all users who have contributed to a specific language
    """
    metadata_df = pd.DataFrame()
    
    # Load metadata for the specified language
    language_path = os.path.join(DATASET_DIR, language, "metadata.csv")
    if not os.path.exists(language_path):
        print(f"No metadata found for language: {language}")
        return
    
    metadata_df = pd.read_csv(language_path)
    
    if not metadata_df.empty:
        # Get unique users who contributed to this language
        user_ids = metadata_df['user_id'].unique()
        print(f"Found {len(user_ids)} users for {language}")
        
        for user_id in user_ids:
            print(f"Sending message to user {user_id}")
            # user_id = _TEST_ID
            try:
                await safe_send_message(bot, user_id, message)
                # Add delay between users
                await asyncio.sleep(2)
            except Forbidden:
                print(f"Error: Bot was blocked by user {user_id}. Skipping this user.")
            except Exception as e:
                print(f"Unexpected error for user {user_id}: {str(e)}")
    else:
        print(f"No users found for language: {language}")

async def send_reminders():
    metadata_df = pd.DataFrame()

    for lang in os.listdir(DATASET_DIR):
        metadata_path = os.path.join(DATASET_DIR, lang, "metadata.csv")
        if os.path.exists(metadata_path):
            lang_df = pd.read_csv(metadata_path)
            lang_df['language'] = lang
            metadata_df = pd.concat([metadata_df, lang_df])

    if not metadata_df.empty:
        user_ids = metadata_df['user_id'].unique()
        
        # Plot progress over time
        plot_progress_over_time()
        
        for user_id in user_ids:
            print(f"Sending reminder to user {user_id}")
            user_langs = metadata_df[metadata_df['user_id'] == user_id]['language'].unique()

            # Build a single message for this user
            message_parts = []
            for lang in user_langs:
                # Get language-specific statistics
                lang_data = metadata_df[metadata_df['language'] == lang]
                total_sentences = lang_data.shape[0]
                total_duration = lang_data['duration'].sum()
                
                # Get user's contribution for this language
                user_lang_data = metadata_df[
                    (metadata_df['user_id'] == user_id) & 
                    (metadata_df['language'] == lang)
                ]
                user_lang_data_count = user_lang_data.shape[0]
                user_duration = user_lang_data['duration'].sum()
                
                lang_display = ' '.join(word.capitalize() for word in lang.split('_'))
                duration_hours = format_duration(total_duration)
                user_duration_hours = format_duration(user_duration)
                
                # Create language-specific message with duration information
                if user_lang_data_count > 30:
                    message_parts.append(
                        f"🙏 You're doing amazing work in {lang_display}! "
                        f"You've contributed {user_lang_data_count:,} sentences "
                        f"({user_duration_hours:.1f} hours). "
                        f"In total, we have {total_sentences:,} sentences "
                        f"({duration_hours:.1f} hours of speech). "
                        f"Can you contribute a bit more? Every sentence helps us improve our model."
                    )
                else:
                    message_parts.append(
                        f"🙏 Your contributions to {lang_display} are invaluable! "
                        f"You've contributed {user_lang_data_count:,} sentences "
                        f"({user_duration_hours:.1f} hours). "
                        f"We've collected {total_sentences:,} sentences in total "
                        f"({duration_hours:.1f} hours of speech). "
                        f"Every sentence you label is a step closer to advancing our "
                        f"speech recognition model. Keep up the amazing work!"
                    )

            # Combine all parts into a single message
            full_message = "\n\n".join(message_parts)
            # user_id = _TEST_ID
            
            try:
                # Send message with retry logic
                await safe_send_message(bot, user_id, full_message)
                # Add a small delay between message and photo
                await asyncio.sleep(1)
                # Send photo with retry logic
                await safe_send_photo(bot, user_id, 'duration_progress_over_time.png')
                # Add delay between users
                await asyncio.sleep(2)
            except Forbidden:
                print(f"Error: Bot was blocked by user {user_id}. Skipping this user.")
            except Exception as e:
                print(f"Unexpected error for user {user_id}: {str(e)}")

    else:
        print("No metadata.csv files found.")

def main():
    parser = argparse.ArgumentParser(description='Send Telegram messages to users')
    parser.add_argument('--language', '-l', help='Target language for the message')
    parser.add_argument('--message', '-m', help='Message to send')
    args = parser.parse_args()

    if args.language and args.message:
        # Send targeted message to users of specific language
        asyncio.run(send_targeted_message(args.language, args.message))
    else:
        # Run the regular reminder function
        asyncio.run(send_reminders())

# Execute the function
if __name__ == "__main__":
    main()