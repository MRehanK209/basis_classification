# k10plus_fetcher/utils.py
import os
import csv
from lxml import etree
import aiohttp
from urllib.parse import urlencode, quote_plus
from pathlib import Path
import logging
import asyncio

# Define namespaces used in picaxml
NAMESPACE = {
    "zs": "http://www.loc.gov/zing/srw/",
    "pica": "info:srw/schema/5/picaXML-v1.0"
}

def create_output_directory(directory_path):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Output directory created at: {directory_path}")
    else:
        logging.info(f"Directory already exists: {directory_path}")

def extract_field(record, tag, code):
    """Extract the first matching subfield from a datafield tag."""
    for field in record.findall(f"pica:datafield[@tag='{tag}']", namespaces=NAMESPACE):
        for subfield in field.findall(f"pica:subfield[@code='{code}']", namespaces=NAMESPACE):
            return subfield.text
    return ""

def extract_multiple_fields(record, tag, code):
    """Extract all matching subfields from a repeated tag (like BK codes)."""
    values = []
    for field in record.findall(f"pica:datafield[@tag='{tag}']", namespaces=NAMESPACE):
        for subfield in field.findall(f"pica:subfield[@code='{code}']", namespaces=NAMESPACE):
            if subfield.text:
                values.append(subfield.text.strip())
    return "|".join(values) if values else ""

def extract_data_from_xml(xml_data):
    """Extract data from the API response XML."""
    # Ensure xml_data is in bytes, not a string.
    if isinstance(xml_data, str):
        xml_data = xml_data.encode("utf-8")

    # Parse the XML data from bytes
    tree = etree.fromstring(xml_data)
    records = []

    for record_data in tree.findall(".//zs:recordData", namespaces=NAMESPACE):
        record = record_data.find("pica:record", namespaces=NAMESPACE)
        if record is None:
            continue
        entry = {
            "PPN": extract_field(record, "003@", "0"),
            "Title": extract_field(record, "021A", "a"),
            "Author": extract_field(record, "028A", "a"),
            "Year": extract_field(record, "011@", "a"),
            "Summary": extract_field(record, "047I", "a"),
            "Work-1": extract_field(record, "039M", "a"),
            "Work-2": extract_field(record, "039M", "a"),
            "Keywords": extract_field(record, "044K", "a"),
            "LOC_Keywords": extract_field(record, "044A", "a"),
            "RVK": extract_multiple_fields(record, "045R", "j"),
            "BK": extract_multiple_fields(record, "045Q", "a")
        }
        records.append(entry)

    return records

def initialize_csv(csv_path, fieldnames):
    """Initialize the CSV file and write the header if it's empty."""
    file_exists = csv_path.exists()
    with open(csv_path, mode='a', newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            logging.info(f"CSV header written for: {csv_path}")

def append_to_csv(csv_path, records):
    """Append extracted records to the CSV file."""
    if records:
        with open(csv_path, mode='a', newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writerows(records)
            logging.info(f"Appended {len(records)} records to: {csv_path}")

async def fetch_data(session, url, start):
    """Fetch data asynchronously."""
    logging.info(f"Fetching records {start} to {start + 99}...")
    async with session.get(url) as response:
        if response.status != 200:
            logging.error(f"Error {response.status}: {await response.text()}")
            return None
        return await response.text()

async def process_batch(session, start, params, csv_file):
    """Process each batch asynchronously."""
    params["startRecord"] = start
    url = f"https://sru.k10plus.de/opac-de-627!rec=1?{urlencode(params, quote_via=quote_plus)}"
    xml_data = await fetch_data(session, url, start)
    if xml_data is None:
        return []

    records = extract_data_from_xml(xml_data)
    append_to_csv(csv_file, records)

    if "<zs:records/>" in xml_data or "<zs:numberOfRecords>0</zs:numberOfRecords>" in xml_data:
        logging.info("No more records.")
        return []

    return records

async def fetch_and_process_all(params, csv_file, year):
    """Main function to fetch all data in parallel for a specific year."""
    async with aiohttp.ClientSession() as session:
        start = 1  # Reset start to 1 for each year
        while True:
            if start > 500000:
                logging.info(f"Breaking the loop for {year} as start value exceeded 500,000.")
                break
            records = await process_batch(session, start, params, csv_file)
            if not records:
                break
            start += params["maximumRecords"]
            await asyncio.sleep(2)  # To prevent overloading the server
