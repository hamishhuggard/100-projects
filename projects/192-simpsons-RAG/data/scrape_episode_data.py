import requests
from bs4 import BeautifulSoup
import time
import re
import json
from datetime import datetime

def is_blank_episode(episode_info, episode_container):
    """
    Determine if an episode is a blank/placeholder episode that shouldn't be included.
    
    Blank episodes typically have:
    1. Generic titles like "Episode #X.Y" 
    2. Future air dates
    3. "Add a plot" button indicating no description
    4. No rating/vote information
    """
    # Check for generic episode title pattern
    title = episode_info.get('episode_title', '')
    if re.match(r'^S\d+\.E\d+\s*∙\s*Episode\s*#\d+\.\d+$', title):
        return True
    
    # Check for "Add a plot" button indicating missing content
    add_plot_button = episode_container.find("a", string=re.compile(r"Add a plot"))
    if add_plot_button:
        return True
    
    # Check if air date is in the future (rough check)
    air_date = episode_info.get('air_date', '')
    if air_date and air_date != 'N/A':
        try:
            # Remove day of week if present (e.g., "Sun, Sep 28, 2025" -> "Sep 28, 2025")
            clean_date = re.sub(r'^[A-Za-z]{3},\s*', '', air_date)
            
            # Try common date formats
            date_formats = [
                '%b %d, %Y',    # Sep 28, 2025
                '%B %d, %Y',    # September 28, 2025
                '%d %b %Y',     # 28 Sep 2025
                '%d %B %Y',     # 28 September 2025
                '%Y-%m-%d',     # 2025-09-28
                '%m/%d/%Y'      # 09/28/2025
            ]
            
            episode_date = None
            for fmt in date_formats:
                try:
                    episode_date = datetime.strptime(clean_date, fmt)
                    break
                except ValueError:
                    continue
            
            if episode_date:
                current_date = datetime.now()
                # If episode is more than 30 days in the future, consider it a placeholder
                if (episode_date - current_date).days > 30:
                    return True
                    
        except Exception as e:
            # If date parsing fails, continue with other checks
            pass
    
    # Check if episode has no description AND no rating
    # This combination usually indicates a placeholder episode
    has_description = episode_info.get('description') not in ['N/A', '', None]
    has_rating = episode_info.get('imdb_rating') not in ['N/A', '', None]
    
    if not has_description and not has_rating:
        # Additional check: if title is very generic or contains placeholder text
        if any(placeholder in title.lower() for placeholder in ['episode #', 'untitled', 'tba', 'to be announced']):
            return True
    
    return False

def scrape_simpsons_episodes():
    """
    Scrapes episode information for 'The Simpsons' from IMDb.
    It iterates through seasons, extracts details for each episode,
    and returns a list of dictionaries containing the scraped data.
    """
    base_url = "https://www.imdb.com/title/tt0096697/episodes/"
    all_episodes = []
    season_number = 1
    # User-Agent to mimic a browser and avoid being blocked by the website
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    # Set an upper limit for seasons to prevent infinite loops in case
    # the "no episodes found" check fails due to unexpected page changes.
    # The user mentioned season 36 is the most recent.
    max_seasons_to_check = 40

    print("Starting IMDb Simpsons episode scraping...")
    print("This process will iterate through each season and includes a delay ")
    print("between requests to be polite to the server. It may take some time.")

    while season_number <= max_seasons_to_check:
        season_url = f"{base_url}?season={season_number}&ref_=ttep"
        print(f"Attempting to fetch data for Season {season_number} from: {season_url}")

        try:
            # Send an HTTP GET request to the season URL
            response = requests.get(season_url, headers=headers)
            # Raise an HTTPError for bad responses (4xx or 5xx)
            response.raise_for_status()
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, "html.parser")

            # Find all episode title elements. These are stable identifiers
            # for individual episode blocks on the page.
            episode_titles_h4 = soup.find_all("h4", attrs={"data-testid": "slate-list-card-title"})

            # If no episode titles are found, it likely means we've gone past the last season
            if not episode_titles_h4:
                print(f"No episode titles found for Season {season_number}. ")
                print("This suggests we have reached the end of available seasons or there are no more episodes.")
                break # Exit the loop

            for title_h4 in episode_titles_h4:
                episode_info = {}
                
                # The main episode container div is the grandparent of the h4 element.
                # This helps to reliably get the full block of information for an episode.
                episode_container = title_h4.find_parent("div").find_parent("div")

                if not episode_container:
                    print(f"Warning: Could not find the main episode container for title: {title_h4.get_text(strip=True)}. Skipping this episode.")
                    continue # Skip to the next episode if container is not found

                # --- Extract Episode Title and URL ---
                title_link_wrapper = title_h4.find("a", class_="ipc-title-link-wrapper")
                if title_link_wrapper:
                    episode_info["season"] = season_number
                    episode_info["episode_title"] = title_link_wrapper.find("div", class_="ipc-title__text--reduced").get_text(strip=True)
                    episode_info["episode_url"] = "https://www.imdb.com" + title_link_wrapper["href"]
                    
                    # Extract the episode number (e.g., S36.E1) from the full title text
                    full_title_text_div = title_h4.find("div", class_="ipc-title__text")
                    if full_title_text_div:
                        full_title_text = full_title_text_div.get_text(strip=True)
                        match = re.search(r'S(\d+)\.E(\d+)', full_title_text)
                        if match:
                            episode_info["episode_number_in_season"] = int(match.group(2))
                        else:
                            episode_info["episode_number_in_season"] = "N/A"
                    else:
                        episode_info["episode_number_in_season"] = "N/A"
                else:
                    episode_info["episode_title"] = "N/A"
                    episode_info["episode_url"] = "N/A"
                    episode_info["episode_number_in_season"] = "N/A"

                # --- Extract Air Date ---
                # The air date span has a dynamic class, but often contains 'knzESm'.
                # We search within the episode container for a span with this partial class.
                air_date_span = episode_container.find("span", class_=re.compile(r"knzESm"))
                if air_date_span:
                    episode_info["air_date"] = air_date_span.get_text(strip=True)
                else:
                    # Fallback: if the dynamic class changes, try to find a date pattern in the container's text
                    date_match = re.search(r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2},\s\d{4}\b', episode_container.get_text())
                    if date_match:
                        episode_info["air_date"] = date_match.group(0)
                    else:
                        episode_info["air_date"] = "N/A"

                # --- Extract Description ---
                description_div = episode_container.find("div", class_="ipc-html-content-inner-div")
                if description_div:
                    episode_info["description"] = description_div.get_text(strip=True)
                else:
                    episode_info["description"] = "N/A"

                # --- Extract IMDb Rating and Vote Count ---
                rating_group = episode_container.find("span", attrs={"data-testid": "ratingGroup--imdb-rating"})
                if rating_group:
                    rating_span = rating_group.find("span", class_="ipc-rating-star--rating")
                    if rating_span:
                        episode_info["imdb_rating"] = rating_span.get_text(strip=True)
                    else:
                        episode_info["imdb_rating"] = "N/A"

                    vote_count_span = rating_group.find("span", class_="ipc-rating-star--voteCount")
                    if vote_count_span:
                        # Clean up vote count text (e.g., "(1.1K)" -> "1.1K", remove non-breaking spaces)
                        votes_text = vote_count_span.get_text(strip=True).replace("(", "").replace(")", "").replace("\u00a0", "")
                        episode_info["vote_count"] = votes_text
                    else:
                        episode_info["vote_count"] = "N/A"
                else:
                    episode_info["imdb_rating"] = "N/A"
                    episode_info["vote_count"] = "N/A"

                # Filter out blank/future episodes
                if not is_blank_episode(episode_info, episode_container):
                    all_episodes.append(episode_info)
                else:
                    print(f"Skipping blank episode: Season {episode_info.get('season', 'N/A')}, Episode {episode_info.get('episode_number_in_season', 'N/A')} - {episode_info.get('episode_title', 'N/A')}")

            season_number += 1
            # Add a delay to prevent overwhelming the server and getting blocked
            time.sleep(1.5)

        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors, especially 404 (Not Found) which might indicate end of seasons
            if e.response.status_code == 404:
                print(f"Season {season_number} page not found (HTTP 404). Assuming end of seasons.")
                break
            else:
                print(f"HTTP Error encountered for Season {season_number}: {e}")
                print("Retrying this season after 5 seconds...")
                time.sleep(5)
                continue # Try fetching the same season again
        except requests.exceptions.RequestException as e:
            # Handle general request errors (e.g., network issues)
            print(f"Network error encountered for Season {season_number}: {e}")
            print("Retrying this season after 5 seconds...")
            time.sleep(5)
            continue # Try fetching the same season again
        except Exception as e:
            # Catch any other unexpected errors
            print(f"An unexpected error occurred while processing Season {season_number}: {e}")
            break # Break the loop on unexpected errors to prevent infinite issues

    print(f"\nScraping complete. Successfully found {len(all_episodes)} episodes across {season_number - 1} seasons.")
    return all_episodes

# Example usage of the scraper function:
if __name__ == "__main__":
    episodes_data = scrape_simpsons_episodes()
    
    # You can now process the 'episodes_data' list.
    # For demonstration, let's print the first 5 episodes and the total count.
    print("\n--- Sample of Scraped Episodes (First 5) ---")
    for i, ep in enumerate(episodes_data[:5]):
        print(f"Episode {i+1}:")
        print(json.dumps(ep, indent=2)) # Pretty print the episode dictionary
        print("-" * 30)
    
    print(f"\nTotal episodes scraped: {len(episodes_data)}")
    
    # You could also save this data to a JSON file:
    # with open("simpsons_episodes.json", "w", encoding="utf-8") as f:
    #     json.dump(episodes_data, f, ensure_ascii=False, indent=4)
    # print("\nData saved to simpsons_episodes.json")

    import csv
    if episodes_data:
        csv_file = "simpsons_episodes.csv"
        keys = episodes_data[0].keys()
        with open(csv_file, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(episodes_data)
        print(f"\nData saved to {csv_file}")