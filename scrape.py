import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path
import re
from bs4 import BeautifulSoup
from bs4.element import Tag
from pandas import Index
from selenium.webdriver.chrome.options import Options
import random
import undetected_chromedriver as uc
from selenium.common.exceptions import TimeoutException
import json
from companies import COMPANIES


LINK_FORMAT = "https://www.levels.fyi/companies/{company}/salaries/software-engineer"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Add more if you want
]


def to_url(company):
    return LINK_FORMAT.format(company=company.lower().replace(" ", "-"))


def scrape_table(driver, xpath):
    table_element = driver.find_element(
        by=By.XPATH,
        value=xpath,
    )
    html = table_element.get_attribute("outerHTML")
    soup = BeautifulSoup(html, "lxml")
    table_tag = soup.find("table")
    if not isinstance(table_tag, Tag):
        return pd.DataFrame()
    headers = [
        th.get_text(strip=True)
        for th in table_tag.find_all("th")
        if isinstance(th, Tag)
    ]
    rows = []
    for tr in table_tag.find_all("tr")[1:] if isinstance(table_tag, Tag) else []:
        tds = (
            [td for td in tr.find_all(["td", "th"]) if isinstance(td, Tag)]
            if isinstance(tr, Tag)
            else []
        )
        row = {}
        for i, td in enumerate(tds):
            col = headers[i] if i < len(headers) else f"col_{i}"
            if col == "Level Name":
                a = td.find("a") if isinstance(td, Tag) else None
                main_name = (
                    a.get_text(strip=True)
                    if isinstance(a, Tag)
                    else td.get_text(strip=True) if isinstance(td, Tag) else str(td)
                )
                span = td.find("span") if isinstance(td, Tag) else None
                subheading = (
                    span.get_text(strip=True) if isinstance(span, Tag) else None
                )
                link = a["href"] if isinstance(a, Tag) and a.has_attr("href") else None
                row["Level Name"] = main_name
                row["Level Subheading"] = subheading
                row["Level Link"] = link
            else:
                row[col] = td.get_text(strip=True) if isinstance(td, Tag) else str(td)
        rows.append(row)
    columns = headers.copy()
    if "Level Name" in columns and "Level Subheading" not in columns:
        columns.insert(columns.index("Level Name") + 1, "Level Subheading")
    if "Level Name" in columns and "Level Link" not in columns:
        columns.insert(columns.index("Level Name") + 1, "Level Link")
    df = pd.DataFrame(rows)
    df = df.reindex(columns=Index(columns))
    return df


def is_usd_df(df):
    for col in df.columns:
        if df[col].astype(str).str.contains(r"\$", na=False).any():
            return True
    return False


def is_usd_csv(path):
    try:
        df = pd.read_csv(path)
        return is_usd_df(df)
    except Exception:
        return False


output_dir = Path("output")
output_dir.mkdir(exist_ok=True)


def get_driver():
    chrome_options = Options()
    chrome_options.add_experimental_option(
        "prefs", {"intl.accept_languages": "en-US,en"}
    )
    chrome_options.add_argument("--incognito")
    # chrome_options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    # chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    driver = uc.Chrome(options=chrome_options)
    driver.delete_all_cookies()
    return driver


def human_motion(driver):
    time.sleep(random.uniform(3, 7))
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
    time.sleep(random.uniform(1, 2))
    driver.execute_script("window.scrollTo(0, 0);")


def scrape():
    for company in COMPANIES:
        output_path = output_dir / f"{company}.csv"
        if output_path.exists() and is_usd_csv(output_path):
            continue
        if output_path.exists():
            output_path.unlink()
        driver = get_driver()
        url = to_url(company=company)
        driver.get(url)
        driver.execute_script("window.localStorage.clear();")
        time.sleep(random.uniform(3, 7))
        xpath = "//div[contains(@class, 'job-family_tableContainer')]/table"
        table = scrape_table(
            driver=driver,
            xpath=xpath,
        )
        if not is_usd_df(table):
            driver.quit()
            raise Exception(
                f"Non-USD currency detected for {company} on initial page load."
            )
        MORE_LEVELS_XPATH = "//a[contains(., 'View') and contains(., 'More Levels')]"
        try:
            button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, MORE_LEVELS_XPATH))
            )
            button.click()
        except TimeoutException:
            pass
        time.sleep(random.uniform(3, 7))
        table = scrape_table(
            driver=driver,
            xpath=xpath,
        )
        if is_usd_df(table):
            table.to_csv(
                output_path,
                index=False,
            )
        human_motion()
        driver.quit()


def combine():
    csv_files = list(output_dir.glob("*.csv"))

    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["Company"] = csv_file.stem
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.to_csv("combined_salaries.csv", index=False)
    print(f"Combined {len(csv_files)} CSV files into combined_salaries.csv")
    print(f"Total rows: {len(combined_df)}")


def extract_number_from_USD(value):
    match = re.search(r"\$([\d,.]+)([KMB]?)", value)
    if not match:
        return 0.0
    number, suffix = match.groups()
    try:
        number = float(number.replace(",", ""))
    except ValueError:
        return 0.0
    if suffix == "K":
        number *= 1_000
    elif suffix == "M":
        number *= 1_000_000
    elif suffix == "B":
        number *= 1_000_000_000
    return number


def extract_of_number(text):
    match = re.search(r"of\s+(\d+)", text)
    if match:
        return float(match.group(1))
    return 0.0


def try_to_click(driver, xpath):
    button = try_to_get_element(driver, xpath)
    if button:
        button.click()


def try_to_get_element(driver, xpath):
    try:
        element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        return element
    except TimeoutException:
        return None


def get_num_results(driver, link):
    driver.get(link)
    num_results_div = try_to_get_element(driver, "//tfoot//div[contains(., 'of')]")
    if num_results_div:
        num_results = extract_of_number(num_results_div.text)
    else:
        num_results = float("-inf")
    return num_results
    # human_motion(driver)


def get_senior_proportions():
    YEAR_THRESHOLDS = [
        0,
        5,
        8,
        10,
    ]
    driver = get_driver()
    # First Startup
    driver.get("https://www.levels.fyi/companies/facebook/salaries/software-engineer")
    try_to_click(
        driver, "//button[contains(text(), 'Added mine already within last 1 year')]"
    )
    try_to_click(driver, "//button[contains(text(), 'Remind Me Later')]")

    csv_files = list(output_dir.glob("*.csv"))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            level_link = row["Level Link"]
            total_compensation = extract_number_from_USD(row["Total"])
            if total_compensation < 500_000 or total_compensation > 1_000_000:
                continue
            QUERY_PARAM_NARROWING = [
                "?yoeChoice=custom&minYoe={yoe}",
                "?yoeChoice=custom&minYoe={yoe}&sinceDate=2-years"
                "?yoeChoice=custom&minYoe={yoe}&sinceDate=2-years&yacChoice=new-only&minYac=0&maxYac=0",
            ]
            i = 0
            qp = 0
            while i < len(YEAR_THRESHOLDS):
                threshold = YEAR_THRESHOLDS[i]
                query_param_narrowing = QUERY_PARAM_NARROWING[qp]
                link = f"https://www.levels.fyi{level_link}{query_param_narrowing.format(yoe=threshold)}"
                colname = f"num_gte_{threshold}_yoe"
                if colname not in df.columns:
                    df[colname] = None
                if df[colname][_] not in [None, float("-inf"), 1]:
                    continue
                num_results = get_num_results(driver, link)
                if num_results == 1_000:
                    qp += 1
                    continue
                df[colname][_] = num_results
                df.to_csv(csv_file, index=False)
                i += 1


def scrape_jsons():
    driver = get_driver()
    for company in COMPANIES:
        out_path = output_dir / f"{company}.json"
        if out_path.exists():
            d = json.load(open(out_path))
            if (
                d["page"] != "/404"
                and d["props"]["pageProps"]["company"]["name"] == company
            ):
                continue
        url = to_url(company=company)
        driver.get(url)

        props_json = try_to_get_element(
            driver, "//script[@type='application/json']"
        ).get_attribute("innerHTML")
        json.dump(json.loads(props_json), open(out_path, "w"), indent=4)
    driver.quit()


def main():
    scrape_jsons()
    # scrape()
    # combine()
    # get_senior_proportions()


if __name__ == "__main__":
    main()
