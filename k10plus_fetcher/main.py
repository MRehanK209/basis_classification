# k10plus_fetcher/main.py
import asyncio
import logging
from pathlib import Path
from utils import create_output_directory, initialize_csv, fetch_and_process_all

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set log level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
    ]
)

# Output directory for all years
out_dir = Path("k10plus_2010_to_2020")
create_output_directory(out_dir)

# SRU API parameters
PARAMS = {
    "version": "1.1",
    "operation": "searchRetrieve",
    "recordSchema": "picaxml",
    "recordPacking": "xml",
    "maximumRecords": 100,
}

# Initialize the CSV file for each year
def initialize_yearly_csv(year):
    """Initialize the CSV file for each year."""
    csv_file = out_dir / f"k10plus_{year}_books.csv"
    initialize_csv(csv_file, fieldnames=["PPN", "Title", "Author", "Year", "Summary", "Work-1", "Work-2", "Keywords", "LOC_Keywords", "RVK", "BK"])
    return csv_file

# Fetch and process data for all years from 2010 to 2020
async def fetch_data_for_all_years():
    """Fetch data for each year from 2010 to 2020."""
    for year in range(2018, 2021):
        logging.info(f"\nStarting to fetch data for {year}...")
        PARAMS["query"] = f"pica.jah={year}"
        csv_file = initialize_yearly_csv(year)
        await fetch_and_process_all(PARAMS, csv_file, year)

# Run the asynchronous fetch and processing for all years
if __name__ == "__main__":
    try:
        asyncio.run(fetch_data_for_all_years())
        logging.info("All records fetched and saved to CSV for all years!")
    except Exception as e:
        logging.error(f"An error occurred during the data fetching process: {e}")
