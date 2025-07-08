import os
import csv
import matplotlib.pyplot as plt
import mplcyberpunk
from datetime import datetime
import json

DATASET_DIR = 'dataset'
DURATION_HISTORY_FILE = 'duration_history.json'

def load_total_duration():
    """
    Load the total duration (in hours) of recorded sentences for each language from metadata files.
    """
    durations = {}
    languages = ["southern_kurdish", "laki_kurdish", "hawrami", "gilaki", 
                 "zazaki", "talysh", "mazanderani", "luri_bakhtiari"]
    
    for lang in languages:
        metadata_path = os.path.join(DATASET_DIR, lang, "metadata.csv")
        total_duration = 0.0
        
        # Read recorded sentence durations from metadata.csv
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as meta_file:
                meta_reader = csv.DictReader(meta_file)
                for row in meta_reader:
                    total_duration += float(row['duration'])  # Assuming 'duration' is in seconds
            durations[lang] = total_duration / 3600  # Convert seconds to hours
        else:
            durations[lang] = 0.0  # Set to 0 if metadata file doesn't exist
    
    return durations

def plot_latest_progress_bar(durations):
    """
    Plot the total duration for each language as a cyberpunk-styled horizontal bar chart, sorted by duration.
    """
    # Sort the durations dictionary by value in descending order
    sorted_durations = {k: v for k, v in sorted(durations.items(), key=lambda item: item[1], reverse=False)}
    
    # Prepare categories and values for the bar plot, and capitalize language names
    categories = [lang.replace('_', ' ').title() for lang in sorted_durations.keys()]
    values = list(sorted_durations.values())
    
    # Define colors for each bar
    colors = [f"C{i % 10}" for i in range(len(categories))]  # Cyclic colors if >10 languages
    
    # Get today's date
    today_date = datetime.today().strftime('%Y-%m-%d')
    
    # Set cyberpunk style
    plt.style.use('cyberpunk')
    
    # Create horizontal bar plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(categories, values, color=colors, zorder=2)
    
    # Add cyberpunk gradient effect
    mplcyberpunk.add_bar_gradient(bars=bars)
    
    # Customize plot
    plt.title(f'Total Duration per Language (Hours)', fontsize=16, pad=20, color='cyan')
    plt.suptitle(f'Date: {today_date}', fontsize=13, color='cyan', y=0.90, ha='center')  # Center the date below the title
    plt.xlabel('Duration (Hours)', labelpad=10, color='cyan', fontsize='xx-large')
    plt.ylabel('Language', labelpad=10, color='cyan', fontsize='xx-large')
    plt.xticks(color='cyan', fontsize='medium', fontweight='bold')
    plt.yticks(color='cyan', fontsize='medium', fontweight='bold')
    
    # Show plot
    plt.tight_layout()
    # plt.show()
    plt.savefig('total_durations_sorted.png', dpi=300)

def load_duration_history():
    """
    Load the existing duration history from the JSON file if it exists.
    """
    if os.path.exists(DURATION_HISTORY_FILE):
        with open(DURATION_HISTORY_FILE, 'r', encoding='utf-8') as json_file:
            return json.load(json_file)
    return {}

def save_duration_to_json(durations):
    """
    Append the durations for each language to the duration_history.json file.
    """
    today_date = datetime.today().strftime('%Y-%m-%d')
    duration_history = load_duration_history()

    # Add today's data to the history
    duration_history[today_date] = durations

    # Save the updated data back to the JSON file
    with open(DURATION_HISTORY_FILE, 'w', encoding='utf-8') as json_file:
        json.dump(duration_history, json_file, ensure_ascii=False, indent=4)
    
    print(f"Duration data appended to {DURATION_HISTORY_FILE}")

def plot_progress_over_time():
    """
    Plot the progress of total duration for each language over time.
    """
    # Load the duration history data
    duration_history = load_duration_history()
    
    # Load total durations and plot the latest progress bar
    durations = load_total_duration()
    if durations:
        plot_latest_progress_bar(durations)
        save_duration_to_json(durations)
    else:
        print("No duration data found.")
    # Prepare data for the plot
    dates = sorted(duration_history.keys())  # Get sorted list of dates
    languages = list(duration_history[next(iter(duration_history))].keys())  # Get language names from any date's data
    
    # Prepare a figure and axis for the plot
    plt.figure(figsize=(12, 8))
    colors = ["#08F7FE", "#FE53BB", "#F5D300", "#00ff41", "#FF0000", "#9467bd", "#66D9EF", "#A6E3FF", "#4EB2FF"]

    
    # Plot progress for each language
    for i, lang in enumerate(languages):
        lang_durations = [duration_history[date].get(lang, 0) for date in dates]  # Get duration for each date for the language
        plt.plot(dates, lang_durations, label=lang.replace('_', ' ').title(), marker='o', color=colors[i], linewidth=2, zorder=1)
    
    # Customize plot
    plt.title('Language Duration Progress Over Time (Hours)', fontsize=16, pad=20, color='cyan')
    plt.xlabel('Date', labelpad=10, color='cyan', fontsize='xx-large')
    plt.ylabel('Duration (Hours)', labelpad=10, color='cyan', fontsize='xx-large')
    plt.xticks(rotation=45, color='cyan', fontsize='medium', fontweight='bold')
    plt.yticks(color='cyan', fontsize='medium', fontweight='bold')
    plt.legend(title='Languages', title_fontsize='15', fontsize='14', loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False)
    
    # Show plot
    plt.tight_layout()
    
    # Add cyberpunk glow effect
    mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.1)

    # plt.show()
    plt.savefig('duration_progress_over_time.png', dpi=300)
    print("Progress over time plot saved as 'duration_progress_over_time.png'")

# Load total durations and plot the latest progress bar
# durations = load_total_duration()
# if durations:
#     plot_latest_progress_bar(durations)
#     save_duration_to_json(durations)
# else:
#     print("No duration data found.")

if __name__ == "__main__":
    # Plot the progress over time
    plot_progress_over_time()


