"""Automated Selenium Browser Verification for Hallucination Verifier UI.

Navigates to the local application in Google Chrome, inputs test claims,
clicks the verify button, waits for live inference completion, extracts
rendered verdicts, badges, and evidence, and captures a browser screenshot.
"""

import os
import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def test_browser_ui():
    chrome_options = Options()
    # Run in headless mode so it can run cleanly and reliably in background,
    # or with standard headless=new
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--window-size=1400,1000")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    print("[1/5] Launching Chrome via Selenium...")
    driver = webdriver.Chrome(options=chrome_options)

    try:
        url = "http://127.0.0.1:8000"
        print(f"[2/5] Navigating to {url}...")
        driver.get(url)

        # Wait for page title and form elements
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "verifyForm"))
        )
        print("Page Title:", driver.title)

        # Inspect initial UI defaults
        mode_select = driver.find_element(By.ID, "mode")
        selected_mode = mode_select.get_attribute("value")
        print("Default Selected Mode in UI:", selected_mode)

        rounds_select = driver.find_element(By.ID, "debateRounds")
        selected_rounds = rounds_select.get_attribute("value")
        print("Default Debate Rounds in UI:", selected_rounds)

        # Input test claim
        test_claim = "Autophagy deficiency leads to the accumulation of damaged mitochondria."
        print(f"[3/5] Entering test passage: '{test_claim}'...")
        textarea = driver.find_element(By.ID, "response")
        textarea.clear()
        textarea.send_keys(test_claim)

        # Click verify button
        verify_btn = driver.find_element(By.ID, "verifyBtn")
        print("[4/5] Clicking 'Verify Response' button...")
        verify_btn.click()

        # Wait for resultsState to become visible (not having 'hidden' class)
        print("Waiting for cross-encoder inference and debate results (up to 60s)...")
        wait = WebDriverWait(driver, 60)
        results_state = wait.until(
            lambda d: d.find_element(By.ID, "resultsState")
            if "hidden" not in d.find_element(By.ID, "resultsState").get_attribute("class")
            else False
        )

        print("[5/5] Results received! Extracting DOM elements...")
        overall_verdict = driver.find_element(By.ID, "overallVerdict").text
        h_index_val = driver.find_element(By.ID, "hIndexValue").text
        conf_score = driver.find_element(By.ID, "confidenceScore").text
        claims_count = driver.find_element(By.ID, "claimsCount").text
        proc_time = driver.find_element(By.ID, "processingTime").text
        u_index_badge = driver.find_element(By.ID, "uIndexBadge").text

        print("=" * 60)
        print("       BROWSER VERIFICATION TEST RESULTS")
        print("=" * 60)
        print(f"Overall Verdict in Browser      : {overall_verdict}")
        print(f"Factual Hallucination Index (H) : {h_index_val}")
        print(f"Coverage Gap Badge (U)          : {u_index_badge}")
        print(f"Calibrated Confidence Score     : {conf_score}")
        print(f"Claims Count                    : {claims_count}")
        print(f"Reported Processing Time        : {proc_time}")

        # Check rendered claims list
        claim_cards = driver.find_elements(By.CSS_SELECTOR, "#claimsList > div")
        print(f"Rendered Claim Cards Count      : {len(claim_cards)}")

        for i, card in enumerate(claim_cards, 1):
            claim_text = card.text.split("\n")[0] if card.text else "N/A"
            print(f"  Claim Card #{i} Summary Preview: {claim_text[:100]}...")

        # Capture browser screenshot
        artifact_dir = r"C:\Users\aravi\.gemini\antigravity\brain\c5d00049-5fd0-4fe0-97d5-fd544e06d1ed"
        os.makedirs(artifact_dir, exist_ok=True)
        screenshot_path = os.path.join(artifact_dir, "browser_verification_screenshot.png")
        driver.save_screenshot(screenshot_path)
        print(f"Saved full browser screenshot to: {screenshot_path}")
        print("=" * 60)

    finally:
        driver.quit()
        print("Chrome browser session closed successfully.")

if __name__ == "__main__":
    test_browser_ui()
