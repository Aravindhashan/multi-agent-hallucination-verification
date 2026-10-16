"""Test Hallucination Detection in Chrome via Selenium."""

import os
import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def test_hallucination_in_browser():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--window-size=1400,1000")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get("http://127.0.0.1:8000")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "verifyForm")))

        # Enter refuted claim
        refuted_claim = "The capital of France is Madrid."
        print(f"Testing Refuted claim: '{refuted_claim}'...")
        textarea = driver.find_element(By.ID, "response")
        textarea.clear()
        textarea.send_keys(refuted_claim)

        # Click verify button
        driver.find_element(By.ID, "verifyBtn").click()

        # Wait for resultsState
        wait = WebDriverWait(driver, 60)
        wait.until(
            lambda d: d.find_element(By.ID, "resultsState")
            if "hidden" not in d.find_element(By.ID, "resultsState").get_attribute("class")
            else False
        )

        overall_verdict = driver.find_element(By.ID, "overallVerdict").text
        h_index_val = driver.find_element(By.ID, "hIndexValue").text
        conf_score = driver.find_element(By.ID, "confidenceScore").text

        print("=" * 60)
        print("    REFUTED CLAIM BROWSER TEST RESULTS")
        print("=" * 60)
        print(f"Overall Verdict in Browser      : {overall_verdict}")
        print(f"Factual Hallucination Index (H) : {h_index_val}")
        print(f"Calibrated Confidence Score     : {conf_score}")

        artifact_dir = r"C:\Users\aravi\.gemini\antigravity\brain\c5d00049-5fd0-4fe0-97d5-fd544e06d1ed"
        screenshot_path = os.path.join(artifact_dir, "browser_refuted_screenshot.png")
        driver.save_screenshot(screenshot_path)
        print(f"Saved refuted browser screenshot to: {screenshot_path}")
        print("=" * 60)

    finally:
        driver.quit()

if __name__ == "__main__":
    test_hallucination_in_browser()
