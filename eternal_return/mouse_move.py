import pyautogui
import time

pyautogui.FAILSAFE = False

seconds = 3_600

for i in range(seconds):
    print(i)
    pyautogui.moveRel(1, 0)
    pyautogui.moveRel(-1, 0)
    time.sleep(1)

print(seconds)
