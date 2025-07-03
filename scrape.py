from pathlib import Path
import json
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

from companies import COMPANIES

LINK_FORMAT = "https://www.levels.fyi/companies/{company}/salaries/software-engineer"

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
    
def to_url(company):
    return LINK_FORMAT.format(company=company.lower().replace(" ", "-"))

def get_driver():
    chrome_options = Options()
    chrome_options.add_experimental_option(
        "prefs", {"intl.accept_languages": "en-US,en"}
    )
    chrome_options.add_argument("--incognito")
    driver = uc.Chrome(options=chrome_options)
    driver.delete_all_cookies()
    return driver

def try_to_get_element(driver, xpath):
    try:
        element = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        return element
    except TimeoutException:
        return None

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

if __name__ == "__main__":
    main()
