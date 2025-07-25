from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import time

options = Options()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("--incognito")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

driver = webdriver.Chrome(options=options)
driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
  "source": """
    Object.defineProperty(navigator, 'webdriver', {
      get: () => undefined
    })
  """
})

driver.get("https://signup.live.com")

try:
    wait = WebDriverWait(driver, 15)
    email_field = wait.until(EC.presence_of_element_located((By.NAME, "MemberName")))
    email_field.send_keys("nomrandom123@undetectedmail.xyz")
    driver.find_element(By.ID, "iSignupAction").click()
except Exception as e:
    print("🔥 ERREUR :", e)

time.sleep(10)
driver.quit()
